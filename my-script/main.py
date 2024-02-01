from src.cyacc import parse


def main():
    while True:
        print(">>", end=" ")
        code = input()
        if code == "exit":
            break
        parse(code)


if __name__ == "__main__":
    main()
