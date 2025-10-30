def main():
    import sys
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    count = 0
    for j in range(N - 2):
        if S[j] == '#' and S[j+1] == '.' and S[j+2] == '#':
            count += 1
    print(count)

if __name__ == "__main__":
    main()