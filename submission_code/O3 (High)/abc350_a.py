def main():
    import sys

    s = sys.stdin.readline().strip()

    # numeric part
    num = int(s[3:])

    if 1 <= num <= 349 and num != 316:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()