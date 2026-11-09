# --- Classes --- 
# Q1
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if len(self.readings) == 0:
            return None  # No readings to calculate an average
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if len(self.readings) == 0:
            return None  # No readings to find the highest
        return max(self.readings)

    def __repr__(self):
        average = self.average()

        return f"Thermometer(location='{self.location}', n_readings={len(self.readings)}, average={average})"


# Create a thermometer
thermometer = Thermometer("Seattle")

# Add four temperature readings
thermometer.add(18)
thermometer.add(22)
thermometer.add(25)
thermometer.add(20)

# Print the results
print("Average temperature:", thermometer.average())
print("Hottest temperature:", thermometer.hottest())


#Q2

# Create the first thermometer
my_thermometer = Thermometer("Charlotte")

my_thermometer.add(15)
my_thermometer.add(18)
my_thermometer.add(20)
my_thermometer.add(21)

# Print the first object
print(my_thermometer)

# Create the second thermometer
another_thermometer = Thermometer("Seattle")

another_thermometer.add(12)
another_thermometer.add(16)
another_thermometer.add(19)

# Print a list containing both objects
print([my_thermometer, another_thermometer])

# Without __repr__, Python displays a default representation
# that usually includes the class name and memory address.
# This is unhelpful for debugging because it does not show
# useful information about the object's data.

#Q3

class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        high_readings = []

        for reading in thermometer.readings:
            if reading > self.threshold:
                high_readings.append(reading)

        return high_readings

my_thermometer = Thermometer("Charlotte")

my_thermometer.add(25)
my_thermometer.add(32)
my_thermometer.add(35)
my_thermometer.add(28)
my_thermometer.add(40)

# Create two alerts with different thresholds
alert1 = TemperatureAlert(30.0)
alert2 = TemperatureAlert(35.0)

# Check the same thermometer with both alerts
print("Readings above 30:", alert1.breaches(my_thermometer))
print("Readings above 35:", alert2.breaches(my_thermometer))

# The threshold is stored in TemperatureAlert so each alert
# can reuse its own temperature limit without passing it every time.
# With twenty thermometers, I can use the same alert to check all
# twenty without repeatedly entering the threshold.


# Dataclasses, Type Hints, and Docstrings
from dataclasses import dataclass
from dataclasses import dataclass, FrozenInstanceError

@dataclass
class Station:
    """Represents a weather station and its location details."""
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float
# Q1

station_a = Station(1, "Seattle Station", 47.6062, -122.3321, 56.0)
station_b = Station(1, "Seattle Station", 47.6062, -122.3321, 56.0)

print(station_a == station_b)

# Output: True
# A dataclass automatically compares the field values of both objects.
# With the original hand-written class, the result would be False
# because regular objects are compared by identity by default.

# Q2
@dataclass(frozen=True)
class Station:
    """Represents a weather station and its location details."""
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station(1, "Seattle Station", 47.6062, -122.3321, 56.0)
station_b = Station(1, "Seattle Station", 47.6062, -122.3321, 56.0)
station_c = Station(2, "Portland Station", 45.5152, -122.6784, 15.0)

# 1. Try to change a field and catch the error.
try:
    station_a.name = "New Station Name"
except FrozenInstanceError as error:
    print(error)

# 2. Create a set containing three stations.
stations = {station_a, station_b, station_c}
print("Set length:", len(stations))

# frozen=True prevents field changes and allows the dataclass to generate
# a hash method. This makes Station objects usable in sets and as dictionary
# keys. Identical stations count as one item in the set.


# Q3
from dataclasses import dataclass, field

@dataclass
class Station:
    """Represents a weather station and its location details."""
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float


@dataclass
class StationBatch:
    """Stores weather stations for a specific region."""
    region: str

    # First try: stations: list[Station] = []
    # Error: ValueError: mutable default <class 'list'> for field
    # stations is not allowed: use default_factory
    # Python refuses because a mutable list could be shared between objects.

    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Adds a station to the batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Returns the highest-elevation station, or None if empty."""
        if len(self.stations) == 0:
            return None

        highest_station = self.stations[0]

        for station in self.stations:
            if station.elevation > highest_station.elevation:
                highest_station = station

        return highest_station


# Create a batch and add stations
batch = StationBatch("Pacific Northwest")

station_a = Station(1, "Seattle", 47.6062, -122.3321, 56.0)
station_b = Station(2, "Mountain Station", 47.5, -121.0, 1200.0)

batch.add(station_a)
batch.add(station_b)

print(batch.highest())

# Test an empty batch
empty_batch = StationBatch("Empty Region")
print(empty_batch.highest())  # Output: None


# Pydantic
# Q1
from pydantic import BaseModel, Field, ValidationError, model_validator


class Reading(BaseModel):
    """Represents a weather reading from a station."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)


# Create a valid Reading object
reading = Reading(
    station_id="SEA01",
    timestamp="2026-10-10 12:00:00",
    temperature_c=18.5,
    humidity=65.0
)

print(reading)

# Q2


class Reading(BaseModel):
    """Represents a weather reading from a station."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)


# 1. Missing required field: humidity
try:
    reading1 = Reading(
        station_id="SEA01",
        timestamp="2026-10-10 12:00:00",
        temperature_c=21.5
    )
except ValidationError as error:
    print("Failure 1: Missing required field")
    print(error)


# 2. Temperature is too high
try:
    reading2 = Reading(
        station_id="SEA01",
        timestamp="2026-10-10 12:00:00",
        temperature_c=150.0,
        humidity=40
    )
except ValidationError as error:
    print("\nFailure 2: Invalid temperature")
    print(error)


# 3. Humidity is not a valid number
try:
    reading3 = Reading(
        station_id="SEA01",
        timestamp="2026-10-10 12:00:00",
        temperature_c=21.5,
        humidity="very humid"
    )
except ValidationError as error:
    print("\nFailure 3: Invalid humidity")
    print(error)


# 4. Pydantic converts compatible values to the expected types
reading4 = Reading(
    station_id="SEA01",
    timestamp="2026-10-10 12:00:00",
    temperature_c="21.5",
    humidity=40
)

print("\nValid reading:")
print(reading4)

print("Temperature type:", type(reading4.temperature_c))
print("Humidity type:", type(reading4.humidity))

# Pydantic converts values when they can be converted to the expected type.
# "21.5" can be converted to a float, but "very humid" cannot.


# Q3
class Reading(BaseModel):
    """Represents a weather reading from a station."""
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)


try:
    reading = Reading(
        station_id="AB",          # Too short
        temperature_c="hot",      # Not a number
        humidity=40
        # timestamp is missing
    )
except ValidationError as e:
    for error in e.errors():
        print("Location:", error["loc"])
        print("Message:", error["msg"])
        print()

    # Three errors are reported: station_id is too short,
    # timestamp is missing, and temperature_c is not numeric.
    # Reporting all errors at once helps us fix multiple problems
    # without rerunning the program after each individual fix.

# Q4

class Reading(BaseModel):
    """Represents a weather reading from a station."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_sensor_failure(self):
        """Reject readings that indicate a possible sensor failure."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Possible sensor failure: humidity is 0 and temperature is below -40°C")
        return self


# 1. Create a valid reading
valid_reading = Reading(
    station_id="SEA01",
    timestamp="2026-10-10 12:00:00",
    temperature_c=21.5,
    humidity=40
)

print("Valid reading:")
print(valid_reading)


# 2. Try the bad combination
try:
    bad_reading = Reading(
        station_id="SEA01",
        timestamp="2026-10-10 12:00:00",
        temperature_c=-45.0,
        humidity=0.0
    )
    print(bad_reading)

except ValidationError as error:
    print("\nInvalid reading:")
    print(error)


# This rule cannot be expressed using Field constraints alone because
# it depends on two fields together: humidity must be 0.0 AND
# temperature_c must be below -40. A model_validator checks both fields.

# pytest
import pytest
# Q1

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

    # pytest.approx is needed because floating-point calculations
    # can produce tiny rounding differences, so exact equality (==)
    # may fail even when the result is practically 98.6.


# Q2
def mean(values: list[float]) -> float:
    """Calculate the average of a list of numbers."""
    if len(values) == 0:
        raise ValueError("Cannot calculate the mean of an empty list.")

    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# Using match="empty" checks the exception message, not just its type.
# Without match=, pytest would pass even if ValueError had the wrong message.

# Q3
print("Running tests with pytest...")
def mean(values: list[float]) -> float:
    """Calculate the average of a list of numbers."""
    if len(values) == 0:
        raise ValueError("Cannot calculate the mean of an empty list.")

    return sum(values) / len(values)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([2.0, 4.0, 6.0], 4.0),
        ([10.0], 10.0),                # Single-element list
        ([-2.0, -4.0], -3.0),          # Negative numbers
        ([1.0, 2.0, 3.0, 4.0], 2.5),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == expected

# Parametrization avoids duplicated test code while checking each case separately.

# collected 6 items                                                                                                                                
# warmup_01.py::test_celsius_to_fahrenheit PASSED                                                                                            [ 16%]
# warmup_01.py::test_mean_of_empty_raises PASSED                                                                                             [ 33%]
# warmup_01.py::test_mean_values[values0-4.0] PASSED                                                                                         [ 50%]
# warmup_01.py::test_mean_values[values1-10.0] PASSED                                                                                        [ 66%]
# warmup_01.py::test_mean_values[values2--3.0] PASSED                                                                                        [ 83%]
# warmup_01.py::test_mean_values[values3-2.5] PASSED                                                                                         [100%]

# =============================================================== 6 passed in 0.11s ================================================================

# Q4
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32  # Deliberately incorrect


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# ==================================================================== FAILURES ====================================================================
# ___________________________________________________________ test_celsius_to_fahrenheit ___________________________________________________________

#     def test_celsius_to_fahrenheit():
#         assert celsius_to_fahrenheit(0) == 32
# >       assert celsius_to_fahrenheit(100) == 212
# E       assert 257.0 == 212
# E        +  where 257.0 = celsius_to_fahrenheit(100)

# warmup_01.py:461: AssertionError
# ============================================================ short test summary info =============================================================
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
# ========================================================== 1 failed, 5 passed in 0.14s ===========================================================