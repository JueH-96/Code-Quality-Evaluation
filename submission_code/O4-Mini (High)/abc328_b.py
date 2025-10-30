def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    D = [int(next(it)) for _ in range(N)]
    ans = 0
    for i in range(1, N+1):
        s = str(i)
        # check if month i is a repdigit (all digits the same)
        if all(ch == s[0] for ch in s):
            d = int(s[0])
            # one‐digit day j = d
            if d <= D[i-1]:
                ans += 1
            # two‐digit day j = dd (e.g. 11, 22, …)
            dd = int(s[0]*2)
            if dd <= D[i-1]:
                ans += 1
    print(ans)

if __name__ == "__main__":
    main()