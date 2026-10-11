from dataclasses import dataclass
from weatherkit.schemas import WeatherResponse

@dataclass
class HourlyReading:
    """Represent one hourly reading.
    Attributes:
        timestamp: Date and time of the observation.
        temperature_c: Temperature in degrees Celsius.
        precipitation_mm: Precipitation in millimeters.
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:

    """Convert a validated weather response into hourly readings.
    Args:
        response: A validated WeatherResponse object.
    Returns:
        A list of HourlyReading objects in the original order.
    """

    readings = []
    hourly = response.hourly
    for i in range(len(hourly.time)):
        reading = HourlyReading(
            timestamp=hourly.time[i],
            temperature_c=hourly.temperature_2m[i],
            precipitation_mm=hourly.precipitation[i],

        )

        readings.append(reading)
    return readings

# WeatherResponse is a Pydantic model because it validates untrusted
# external API data at the system boundary. HourlyReading is a dataclass
# because it represents clean, already-validated internal data.