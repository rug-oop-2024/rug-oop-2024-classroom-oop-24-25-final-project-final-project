from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """Artifact.
    Atributes:
        name (str): name of artifect.
        asset_path (str): path of the asset.
        data (bytes): base64 encoded path to the data.
        version (str): version of the artifact.
        type (str): type of artifact
    """
    name: str = Field()
    asset_path: str = Field()
    data: bytes = Field()
    version: str = Field()
    type: str = Field()

    def read(self) -> bytes:
        return self.data
