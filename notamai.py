import datetime
import logging
import sys
from typing import List, Optional, Union

import httpx
import instructor
from geographiclib.constants import Constants
from geographiclib.geodesic import Geodesic
from openai import OpenAI
from pydantic import BaseModel, Field, field_serializer

logger = logging.getLogger(__name__)


class Error(Exception):
    pass


def parse_coords(coords: str) -> tuple[float, float]:
    """Converts a string like '422750N1154403W' to a decimal lon,lat like
    [-115.734, 42.463]."""
    lat = float(coords[0:2]) + float(coords[2:4]) / 60 + float(coords[4:6]) / 3600
    lon = float(coords[7:10]) + float(coords[10:12]) / 60 + float(coords[12:14]) / 3600
    if coords[6] == "S":
        lat = -lat
    if coords[14] == "W":
        lon = -lon
    return (lon, lat)


def nautical_miles_to_meters(nautical_miles: float) -> float:
    return 1852.0 * nautical_miles


def get_endpoint(
    lat1: float, lon1: float, bearing_deg: float, dist_m: float
) -> tuple[float, float]:
    "Given a start point, bearing, and distance, returns the end point."
    geod = Geodesic(Constants.WGS84_a, Constants.WGS84_f)
    d = geod.Direct(lat1, lon1, bearing_deg, dist_m)
    if lon1 < 0 and d["lon2"] > 0:
        d["lon2"] = -179.99
    return d["lon2"], d["lat2"]


def create_circle_polygon(
    center: tuple[float, float], radius_m: float, num_segments: int
) -> list[tuple[float, float]]:
    circle_polygon = []
    bearing_step = 360.0 / num_segments
    bearing = 0.0
    while bearing < 360.0:
        circle_polygon.append(get_endpoint(center[1], center[0], bearing, radius_m))
        bearing += bearing_step
    circle_polygon.append(circle_polygon[0])
    return circle_polygon


class SurfaceAlt(BaseModel):
    """SFC AKA surface."""

    type: str = "SFC"


class UnlimitedAlt(BaseModel):
    """UNL AKA unlimited altitude."""

    type: str = "UNL"


class MslAlt(BaseModel):
    """MSL AKA mean sea level altitude."""

    type: str = "MSL"
    height_ft: int


class AglAlt(BaseModel):
    """AGL AKA above ground level altitude."""

    type: str = "AGL"
    height_ft: int


class FlAlt(BaseModel):
    """Flight level altitude, e.g. 'FL190'. Height units are flight levels, not feet."""

    type: str = "FL"
    height_ft: int = Field(
        default=None,
        description="Flight level, e.g. for 'FL350' flight_level=350. 'FL200' -> flight_level=200.",
    )


class AltitudeRange(BaseModel):
    """
    Represents an altitude range or vertical limits. Must have min and max.
    """

    type: str = "RANGE"
    min: Union[SurfaceAlt, UnlimitedAlt, MslAlt, AglAlt, FlAlt]
    max: Union[SurfaceAlt, UnlimitedAlt, MslAlt, AglAlt, FlAlt]


class Coordinates(BaseModel):
    """
    Always convert lat and lon to this format: DDMMSS[N|S] and DDDMMSS[W|E],
    e.g. 422750N and 1154403W.
    """

    lat: str
    lon: str


class RangeRing(BaseModel):
    center: Coordinates
    radius_nm: float
    altitude: Union[MslAlt, AglAlt, SurfaceAlt, AltitudeRange, UnlimitedAlt, FlAlt]


class Polygon(BaseModel):
    coordinates: List[Coordinates]
    altitude: AltitudeRange


def convert_datetime_to_iso_8601_with_z_suffix(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def transform_to_utc_datetime(dt: datetime.datetime) -> datetime.datetime:
    return dt.astimezone(tz=datetime.timezone.utc)


class Notam(BaseModel):
    """Represents an FAA NOTAM."""

    accountability: Optional[str] = Field(
        default=None,
        description="Usually a 3-letter code. Often an ARTCC code like 'ZLC', an airport code, or 'GPS'.",
    )
    location: Optional[str]
    number: Optional[str] = Field(
        default=None,
        description="The NOTAM number, e.g. '01/020'.",
    )
    description: Optional[str]
    range_rings: List[RangeRing]
    polygons: List[Polygon]
    daily_times: List[str] = Field(
        default_factory=list,
        description="List of the daily times the NOTAM is active, in Zulu time. E.g. 1000Z-1300Z.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="The start date and time of the NOTAM in ISO-8601 format, UTC time. Optional.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="The end date and time of the NOTAM in ISO-8601 format, UTC time. Optional.",
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="List of any caveats or additional information about the NOTAM.",
    )
    
    def as_geojson(self) -> dict:
        """Returns a GeoJSON FeatureCollection."""
        feature_collection = {
            "type": "FeatureCollection",
            "features": [],
        }
        if self.range_rings:
            for range_ring in self.range_rings:
                feature_collection["features"].append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                create_circle_polygon(
                                    parse_coords(
                                        range_ring.center.lat + range_ring.center.lon
                                    ),
                                    nautical_miles_to_meters(range_ring.radius_nm),
                                    300,
                                )
                            ],
                        },
                        "properties": {
                            "number": self.number,
                            "title": self.description,
                            "accountability": self.accountability,
                            "start_date": str(self.start_date),
                            "end_date": str(self.end_date),
                            "daily_times": self.daily_times,
                            "altitude": str(range_ring.altitude),
                        },
                    }
                )
        if self.polygons:
            for polygon in self.polygons:
                feature_collection["features"].append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    parse_coords(coord.lat + coord.lon)
                                    for coord in polygon.coordinates
                                ]
                            ],
                        },
                        "properties": {
                            "number": self.number,
                            "title": self.description,
                            "accountability": self.accountability,
                            "start_date": str(self.start_date),
                            "end_date": str(self.end_date),
                            "daily_times": self.daily_times,
                            "altitude": str(polygon.altitude),
                        },
                    }
                )
        return feature_collection


MaybeNotam = instructor.Maybe(Notam)


NOTAM1_TXT = """!GPS 01/020 ZLC NAV GPS (MHRC GPS 24-02)(INCLUDING WAAS,GBAS, AND 
ADS-B) MAY NOT BE AVBL WI A 372NM RADIUS CENTERED AT
424006N1153225W(TWF266048) FL400-UNL,
332NM RADIUS AT FL250,
229NM RADIUS AT 10000FT,
233NM RADIUS AT 4000FT AGL,
193NM RADIUS AT 50FT AGL.
DLY 1700-2000
2401081700-2401122000"""

NOTAM1 = Notam(
    number="01/020",
    accountability="GPS",
    location="ZLC",
    description="GPS (MHRC GPS 24-02)(INCLUDING WAAS,GBAS, AND ADS-B) MAY NOT BE AVBL",
    start_date='2021-01-08T17:00:00Z',
    end_date='2021-01-12T20:00:00Z',
    daily_times=["1700-2000"],
    range_rings=[
        RangeRing(
            center=Coordinates(lat="424006N", lon="1153225W"),
            radius_nm=372,
            altitude=AltitudeRange(min=FlAlt(height_ft=400), max=UnlimitedAlt()),
        ),
        RangeRing(
            center=Coordinates(lat="424006N", lon="1153225W"),
            radius_nm=332,
            altitude=FlAlt(height_ft=250),
        ),
        RangeRing(
            center=Coordinates(lat="424006N", lon="1153225W"),
            radius_nm=229,
            altitude=MslAlt(height_ft=10000),
        ),
        RangeRing(
            center=Coordinates(lat="424006N", lon="1153225W"),
            radius_nm=233,
            altitude=AglAlt(height_ft=4000),
        ),
        RangeRing(
            center=Coordinates(lat="424006N", lon="1153225W"),
            radius_nm=193,
            altitude=AglAlt(height_ft=50),
        ),
    ],
    polygons=[],
    caveats=[],
)

NOTAM2_TXT = """!GPS 01/013 ZLA NAV GPS (SCTTR GPS 24-02) (INCLUDING WAAS, GBAS, AND
ADS-B) MAY NOT BE AVBL WI AN AREA DEFINED AS 280835N1221912W TO
270822N1164850W TO 301204N1132704W TO 333917N1182158W TO
332220N1200359W TO 280835N1221912W
SFC-UNL
DLY 0900-2300
2401081700-2401192200"""

NOTAM2 = Notam(
    number="01/013",
    accountability="GPS",
    location="ZLA",
    description="GPS (SCTTR GPS 24-02) (INCLUDING WAAS, GBAS, AND ADS-B) MAY NOT BE AVBL",
    start_date='2021-01-08T17:00:00Z',
    end_date='2021-01-19T22:00:00Z',
    daily_times=["0900-2300"],
    range_rings=[],
    polygons=[
        Polygon(
            coordinates=[
                Coordinates(lat="280835N", lon="1221912W"),
                Coordinates(lat="270822N", lon="1164850W"),
                Coordinates(lat="301204N", lon="1132704W"),
                Coordinates(lat="333917N", lon="1182158W"),
                Coordinates(lat="332220N", lon="1200359W"),
                Coordinates(lat="280835N", lon="1221912W"),
            ],
            altitude=AltitudeRange(min=SurfaceAlt(), max=UnlimitedAlt()),
        )
    ],
    caveats=[],
)

NOTAM3_TXT = """!GPS 05/067 ZLA NAV GPS (SCTTR GPS 24-03) (INCLUDING
WAAS, GBAS, AND ADS-B) MAY NOT BE AVBL WI AN AREA DEFINED AS:
282847N1212520W TO 274903N1164760W TO 301046N1134113W TO
335522N1182859W TO 334320N1192202W TO 282847N1212520W, SFC-FL400.

2405140700-2405141300"""

NOTAM3 = Notam(
    number="05/067",
    accountability="GPS",
    location="ZLA",
    description="GPS (SCTTR GPS 24-03) (INCLUDING WAAS, GBAS, AND ADS-B) MAY NOT BE AVBL",
    start_date='2024-05-14T07:00:00Z',
    end_date='2024-05-14T13:00:00Z',
    daily_times=[],
    polygons=[
        Polygon(
            coordinates=[
                Coordinates(lat="282847N", lon="1212520W"),
                Coordinates(lat="274903N", lon="1164760W"),
                Coordinates(lat="301046N", lon="1134113W"),
                Coordinates(lat="335522N", lon="1182859W"),
                Coordinates(lat="334320N", lon="1192202W"),
                Coordinates(lat="282847N", lon="1212520W"),
            ],
            altitude=AltitudeRange(min=SurfaceAlt(), max=FlAlt(height_ft=400)),
        )
    ],
    range_rings=[],
    caveats=[],
)

NOTAM4_TXT = """!GPS 05/071 ZLA NAV GPS (SCTTR GPS 24-03) (INCLUDING
WAAS, GBAS, AND ADS-B) MAY NOT BE AVBL WI AN AREA DEFINED AS:
282847N1212520W TO 274903N1164760W TO 301046N1134113W TO
335522N1182859W TO 334320N1192202W TO 282847N1212520W, SFC-UNL.
0700Z-1300Z
2405150700-2405151300"""

NOTAM4 = Notam(
    number="05/071",
    accountability="GPS",
    location="ZLA",
    description="GPS (SCTTR GPS 24-03) (INCLUDING WAAS, GBAS, AND ADS-B) MAY NOT BE AVBL",
    start_date='2024-05-15T07:00:00Z',
    end_date='2024-05-15T13:00:00Z',
    daily_times=["0700Z-1300Z"],
    polygons=[
        Polygon(
            coordinates=[
                Coordinates(lat="282847N", lon="1212520W"),
                Coordinates(lat="274903N", lon="1164760W"),
                Coordinates(lat="301046N", lon="1134113W"),
                Coordinates(lat="335522N", lon="1182859W"),
                Coordinates(lat="334320N", lon="1192202W"),
                Coordinates(lat="282847N", lon="1212520W"),
            ],
            altitude=AltitudeRange(min=SurfaceAlt(), max=UnlimitedAlt()),
        )
    ],
    range_rings=[],
    caveats=[],
)

NOTAM5_TXT = """!GPS 04/043 ZLA NAV GPS (WSMRNM GPS 24-30) (INCLUDING WAAS, GBAS,
AND ADS-B) MAY NOT BE AVBL WI A 365NM RADIUS CENTERED AT
333101N1063525W (TCS055037) FL400-UNL,
309NM RADIUS AT FL250
233NM RADIUS AT 10000FT,
213NM RADIUS AT 4000FT AGL,
185NM RADIUS AT 50FT AGL.
DLY 1830-2230
2404121830-2404142230"""

NOTAM5 = Notam(
    number="04/043",
    accountability="GPS",
    location="ZLA",
    description="GPS (WSMRNM GPS 24-30) (INCLUDING WAAS, GBAS, AND ADS-B) MAY NOT BE AVBL",
    start_date='2024-04-12T18:30:00Z',
    end_date='2024-04-14T22:30:00Z',
    daily_times=["1830-2230"],
    range_rings=[
        RangeRing(
            center=Coordinates(lat="333101N", lon="1063525W"),
            radius_nm=365,
            altitude=AltitudeRange(min=FlAlt(height_ft=400), max=UnlimitedAlt()),
        ),
        RangeRing(
            center=Coordinates(lat="333101N", lon="1063525W"),
            radius_nm=309,
            altitude=FlAlt(height_ft=250),
        ),
        RangeRing(
            center=Coordinates(lat="333101N", lon="1063525W"),
            radius_nm=233,
            altitude=MslAlt(height_ft=10000),
        ),
        RangeRing(
            center=Coordinates(lat="333101N", lon="1063525W"),
            radius_nm=213,
            altitude=AglAlt(height_ft=4000),
        ),
        RangeRing(
            center=Coordinates(lat="333101N", lon="1063525W"),
            radius_nm=185,
            altitude=AglAlt(height_ft=50),
        ),
    ],
    polygons=[],
    caveats=[],
)

NOTAM6_TXT = """ZIT CEREMONIE
Lateral limits
Circle of 80 NM radius centred on 48°51'11"N - 002°21'00"E
Vertical Limits
SFC/UNL"""

NOTAM6 = Notam(
    number=None,
    accountability="ZIT",
    location=None,
    description="CEREMONIE",
    range_rings=[
        RangeRing(
            center=Coordinates(lat="485111N", lon="0022100E"),
            radius_nm=80,
            altitude=AltitudeRange(min=SurfaceAlt(), max=UnlimitedAlt()),
        )
    ],
    polygons=[],
    start_date=None,
    end_date=None,
    caveats=[],
)


NOTAM7_TXT = """ZIT CEREMONIE
Lateral limits
Circle of 80 NM radius centred on 48°51'11"N - 002°21'00"E
-the areas LE-P1, LF-P3, LF-P4, LF-P6, LF-P1, LF-P34, LF-PA6, IF. LF=P52, LF-P65, LF-P6, LF-P6, LF-P75, LF-P82, LF-P/25.,
- the interfering portions of the areas LF-P27 and LF-P33.
Vertical Limits
SFC/UNL"""

NOTAM7 = Notam(
    number=None,
    accountability="ZIT",
    location=None,
    description="CEREMONIE",
    range_rings=[
        RangeRing(
            center=Coordinates(lat="485111N", lon="0022100E"),
            radius_nm=120,
            altitude=AltitudeRange(min=SurfaceAlt(), max=UnlimitedAlt()),
        )
    ],
    polygons=[],
    start_date=None,
    end_date=None,
    caveats=[
        "The areas LE-P1, LF-P3, LF-P4, LF-P6, LF-P1, LF-P34, LF-PA6, IF. LF=P52, LF-P65, LF-P6, LF-P6, LF-P75, LF-P82, LF-P/25 are unknown.",
    ],
)

NOTAM8_TXT = """A2734/24 NOTAMN
Q) OIIX/QWMLW/IV/BO/W/000/120/
A) OIIX B) 2408071130 C) 2408080730
D) AUG 07 /1130-1430
AUG 08 /0430-0730
E) GUN FIRING WILL TAKE PLACE WI AREA :
342123N 0495625E
342604N 0500121E
342300N 0500706E
341805N 0500522E
F) GND G) 12000FT AMSL
CREATED: 06 Aug 2024 08:03:00
SOURCE: OIIIYNYX"""

NOTAM8 = Notam(
    number="A2734/24",
    accountability="OIIX",
    location="OIIX",
    description="GUN FIRING WILL TAKE PLACE",
    start_date='2024-08-07T11:30:00Z',
    end_date='2024-08-08T07:30:00Z',
    daily_times=["1130-1430", "0430-0730"],
    range_rings=[],
    polygons=[
        Polygon(
            coordinates=[
                Coordinates(lat="342123N", lon="0495625E"),
                Coordinates(lat="342604N", lon="0500121E"),
                Coordinates(lat="342300N", lon="0500706E"),
                Coordinates(lat="341805N", lon="0500522E"),
                Coordinates(lat="342123N", lon="0495625E"),
            ],
            altitude=AltitudeRange(min=SurfaceAlt(), max=MslAlt(height_ft=12000)),
        )
    ],
    caveats=[],
)

NOTAM9_TXT = """A0451/24 NOTAMN
Q) LLLL/QANLT/I /NBO/E /110/600/3226N03421E010
A) LLLL B) 2407010001 C) 2408312059 E) ATS RTE W13 AVBL BTN JILET-DAFNA, BTN 11,000FT-16,000FT AMSL
H24 AND BTN TAPUZ-DAFNA BTN FL330-FL600, AS FLW:
SUN 2000 - MON 0400
MON 2000 - TUE 0400
TUE 2000 - WED 0400
WED 2000 - THU 0400
THU 1400 - SUN 0500
ALL TIMES UTC.
CREATED: 23 Jun 2024 11:46:00
SOURCE: EUECYTYN
"""

NOTAM9 = Notam(
    number="A0451/24",
    accountability="LLLL",
    location="LLLL",
    description="ATS RTE W13 AVBL BTN JILET-DAFNA, BTN 11,000FT-16,060FT AMSL H24 AND BTN TAPUZ-DAFNA BTN FL330-FL600",
    start_date='2024-07-01T00:01:00Z',
    end_date='2024-08-31T20:59:00Z',
    daily_times=[
        "SUN 2000 - MON 0400",
        "MON 2000 - TUE 0400",
        "TUE 2000 - WED 0400",
        "WED 2000 - THU 0400",
        "THU 1400 - SUN 0500",
    ],
    range_rings=[
        RangeRing(
            center=Coordinates(lat="322600N", lon="0342100E"),
            radius_nm=10,
            altitude=AltitudeRange(min=MslAlt(height_ft=11000), max=MslAlt(height_ft=16000)),
        )
    ],
    polygons=[],
    caveats=[],
)

def few_shot_messages() -> List[dict]:
    messages = []
    for text, result in [
        (NOTAM1_TXT, NOTAM1),
        (NOTAM2_TXT, NOTAM2),
        (NOTAM3_TXT, NOTAM3),
        (NOTAM4_TXT, NOTAM4),
        (NOTAM6_TXT, NOTAM6),
        (NOTAM7_TXT, NOTAM7),
        (NOTAM8_TXT, NOTAM8),
    ]:
        messages.append(
            {"role": "user", "content": "Decode the following NOTAM:\n\n" + text}
        )
        messages.append(
            {
                "role": "function",
                "name": "Notam",
                "content": result.model_dump_json(),
            }
        )
    return messages


def parse_notam(openai_model: str, notam_txt: str) -> Notam:
    client = instructor.from_openai(
        OpenAI(
            http_client=httpx.Client(
                event_hooks={
                    "response": [],
                    "request": [],
                }
            )
        )
    )
    messages = []
    messages += few_shot_messages()
    messages.append(
        {"role": "user", "content": "Decode the following NOTAM:\n\n" + notam_txt}
    )
    # Complete with a few-shot prompt.
    notam = client.chat.completions.create(
        # model="gpt-3.5-turbo-1106",
        model=openai_model,
        messages=messages,
        response_model=Notam,
        temperature=0.0,
    )
    return notam


def parse_notam_streaming(openai_model: str, notam_txt: str):
    client = instructor.from_openai(
        OpenAI(
            http_client=httpx.Client(
                event_hooks={
                    "response": [],
                    "request": [],
                }
            )
        )
    )
    messages = []
    messages.append(
        {
            "role": "system",
            "content": (
                "You are an expert at decoding FAA NOTAMS. You're "
                "fluent in the language of NOTAMs, e.g. 'FL200' is "
                "the altitude flight level 200."
            ),
        }
    )
    messages += few_shot_messages()
    messages.append(
        {"role": "user", "content": "Decode the following NOTAM:\n\n" + notam_txt}
    )
    stream = client.chat.completions.create_partial(
        model=openai_model,
        messages=messages,
        response_model=Notam,
        stream=True,
        temperature=0.0,
    )
    for partial in stream:
        yield partial


def main():
    logging.basicConfig(level=logging.INFO)
    notam_txt = sys.stdin.read()
    notam = parse_notam(notam_txt)
    print(notam.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
