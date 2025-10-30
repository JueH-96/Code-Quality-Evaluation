def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    S = input().rstrip()
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            print(i + 1)
            return
    print(-1)

if __name__ == "__main__":
    main()