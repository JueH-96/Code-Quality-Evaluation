def main():
    import sys
    input = sys.stdin.readline

    N, K = map(int, input().split())
    S = input().strip()

    ans = 0
    cnt = 0
    for c in S:
        if c == 'O':
            cnt += 1
        else:
            ans += cnt // K
            cnt = 0
    ans += cnt // K

    print(ans)

if __name__ == "__main__":
    main()