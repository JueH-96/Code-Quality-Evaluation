def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    count = 0
    # Check every position i (0-based) for pattern # . #
    for i in range(N - 2):
        if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
            count += 1
    print(count)

if __name__ == "__main__":
    main()