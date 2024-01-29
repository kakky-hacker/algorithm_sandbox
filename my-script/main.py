from src.cyacc import parse


def main():
    with open("sample/code.txt", "r") as f:
        code = f.read()
        for data in code.split("\n"):
            if data != "":
                parse(data)


if __name__ == "__main__":
    main()
