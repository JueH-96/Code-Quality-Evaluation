def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    D = list(map(int, data[1:]))

    ans = 0
    for i in range(1, N+1):
        s = str(i)
        # check if month number i is a repdigit
        if all(ch == s[0] for ch in s):
            d = s[0]
            # count days j that are repdigits of the same digit d
            # up to 3-digit repdigits are enough since D[i-1] <= 100
            for k in range(1, 4):
                j = int(d * k)
                if j <= D[i-1]:
                    ans += 1
    print(ans)

if __name__ == "__main__":
    main()