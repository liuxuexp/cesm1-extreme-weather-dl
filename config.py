import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.environ.get(
    "RAW_DATA_DIR",
    os.path.join(BASE_DIR, "data", "raw")
)

LABELED_DATA_DIR = os.path.join(BASE_DIR, "data", "labeled")

MASK_DIR = os.path.join(BASE_DIR, "data", "land_sea_mask")

LAND_SEA_MASK = os.environ.get(
    "LAND_SEA_MASK",
    os.path.join(MASK_DIR, "Land-sea-mask-0.75.nc")
)

RESULTS_SUMMER_DIR = os.path.join(BASE_DIR, "results", "summer")
RESULTS_WINTER_DIR = os.path.join(BASE_DIR, "results", "winter")

OUTPUT_DIR = BASE_DIR

SEASONS = {
    "s": {
        "name": "summer",
        "savefile": "combine-summer-1.nc",
        "savefilez": "combine-summerz-1.nc",
        "labeled_t2m": "summer-t2m-t2man-labled-c5-7.nc",
        "labeled_z500": "summer-z500-labled-c5-7.nc",
        "fd": 151 - 15,
        "ld": 242 + 15,
        "nyears": 86,
    },
    "w": {
        "name": "winter",
        "savefile": "combine-winter-1.nc",
        "savefilez": "combine-winterz-1.nc",
        "labeled_t2m": "winter-t2m-t2man-labled-c5-7.nc",
        "labeled_z500": "winter-z500-labled-c5-7.nc",
        "fd": 334 - 15 - 181,
        "ld": 423 + 15 - 181,
        "nyears": 85,
    },
}

SPATIAL_DOMAIN = {
    "t2m": {"lat_north": 55, "lat_south": 20, "lon_west": 70, "lon_east": 135},
    "z500": {"lat_north": 90, "lat_south": 10, "lon_west": 70, "lon_east": 135},
}

N_MEMBERS = 42
N_CLUSTERS = 4
NORMAL_DAY_NUM = 800
ROLLING_MEAN_WINDOW = 15
PERCENTILE_THRESHOLD = 99
CONSECUTIVE_DAYS_FOR_EXTREME = 5
EVENT_SEPARATION_DAYS = 16
LEAD_LAG_DAYS = 6
