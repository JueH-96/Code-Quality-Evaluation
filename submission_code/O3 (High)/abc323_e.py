import sys

MOD = 998244353


def main() -> None:
    input = sys.stdin.readline

    N, X = map(int, input().split())
    T = list(map(int, input().split()))

    invN = pow(N, MOD - 2, MOD)          # modular inverse of N
    dp = [0] * (X + 1)                   # dp[s] … probability that the
                                         # current total length is exactly s
                                         # and we have never exceeded X yet
    dp[0] = 1

    for s in range(X + 1):
        if dp[s] == 0:
            continue
        add = dp[s] * invN % MOD         # probability weight for each next song
        for l in T:
            nxt = s + l
            if nxt <= X:                 # still not past the border → keep going
                dp[nxt] = (dp[nxt] + add) % MOD

    # finish with Song 1 (index 0) and overshoot X
    ans = 0
    length_1 = T[0]
    for s in range(X + 1):
        if dp[s] and s + length_1 > X:   # crossing the border with Song 1
            ans = (ans + dp[s] * invN) % MOD

    print(ans)


if __name__ == "__main__":
    main()