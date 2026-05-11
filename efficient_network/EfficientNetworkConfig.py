from typing import List


class EfficientNetworkConfig:
    def __init__(self, variant: str):
        if variant in ["small", "s"]:
            self.__config = [
                (2, 24, 24, 1, 1, True),
                (4, 24, 48, 4, 2, True),
                (4, 48, 64, 4, 2, True),
                (6, 64, 128, 4, 2, False),
                (9, 128, 160, 6, 1, False),
                (15, 160, 256, 6, 2, False),
            ]

    def get(self) -> List[tuple]:
        return self.__config
