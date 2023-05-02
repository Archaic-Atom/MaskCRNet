# -*coding: utf-8 -*-
import JackFramework as jf
from UserModelImplementation.user_interface import UserInterface


def main() -> None:
    app = jf.Application(UserInterface(), "Stereo Mtahcing Network for Remote Sensing")
    app.start()


# execute the main function
if __name__ == "__main__":
    main()
