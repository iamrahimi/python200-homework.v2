from pydantic import BaseModel, Field, model_validator

class HourlyBlock(BaseModel):

    """Validate the hourly weather measurements returned by the API."""
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]
    @model_validator(mode="after")
    def check_list_lengths(self):
        """Ensure all hourly measurement lists have equal lengths."""
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)
        ):
            raise ValueError(
                "time, temperature_2m, and precipitation "
                "must have the same length"
            )
        return self

class WeatherResponse(BaseModel):
    """Validate the top-level weather API response."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock