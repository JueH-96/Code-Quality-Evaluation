def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    # First line is the integer N (not used explicitly since we can work with S directly)
    N = int(input_data[0].strip())
    S = input_data[1].strip()

    # Using str.find to obtain the first index of "ABC" (0-indexed)
    pos = S.find("ABC")
    if pos == -1:
        print(-1)
    else:
        # Adjusting to 1-indexed position
        print(pos + 1)

if __name__ == '__main__':
    main()