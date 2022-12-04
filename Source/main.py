# -*coding: utf-8 -*-
import JackFramework as jf
from UserModelImplementation.user_interface import UserInterface


def main() -> None:
    app = jf.Application(UserInterface(), "your network name")
    app.start()


# execute the main function
if __name__ == "__main__":
    main()
