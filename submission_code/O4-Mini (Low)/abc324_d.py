import sys
import math

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    S = input().strip()
    # Count digits in S
    target = [0]*10
    for c in S:
        target[ord(c)-48 if False else int(c)] += 1

    # Maximum square root such that k^2 <= 10^N - 1
    max_k = math.isqrt(10**N - 1)

    ans = 0
    # For each k from 0 to max_k, form its square with leading zeros to length N
    for k in range(max_k+1):
        sq = k*k
        s = str(sq).zfill(N)
        # quick prune: if first digit is not in target and that digit count is zero, skip
        # but we'll just count directly
        cnt = [0]*10
        for ch in s:
            cnt[ord(ch)-48 if False else int(ch)] += 1
        if cnt == target:
            ans += 1

    print(ans)

if __name__ == "__main__":
    main()