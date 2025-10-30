def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    pos = S.find("ABC")
    if pos == -1:
        print(-1)
    else:
        # convert 0-based to 1-based index
        print(pos + 1)

if __name__ == "__main__":
    main()