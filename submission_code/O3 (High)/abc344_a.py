def main():
    import sys

    S = sys.stdin.readline().rstrip()

    first = S.find('|')
    second = S.find('|', first + 1)  # find next '|' after the first

    result = S[:first] + S[second + 1:]
    print(result)


if __name__ == "__main__":
    main()