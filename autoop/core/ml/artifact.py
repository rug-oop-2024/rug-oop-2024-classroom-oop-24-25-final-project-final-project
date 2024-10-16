from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    name: str = Field()
    asset_path: str = Field()
    data: bytes = Field()
    version: str = Field()
    type: str = Field()

    def read(self) -> bytes:
        return self.data
