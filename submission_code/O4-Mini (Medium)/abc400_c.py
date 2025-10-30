import sys
import math

def main():
    data = sys.stdin.read().split()
    N = int(data[0])

    ans = 0
    # We want all X = 2^v * (t^2) <= N
    # where v >= 1 and t is odd.
    # For each v, count odd t with t^2 <= N/2^v.
    v = 1
    while (1 << v) <= N:
        M = N >> v             # floor(N / 2^v)
        s = math.isqrt(M)      # max t with t^2 <= M
        # count odd t in [1..s] is ceil(s/2) = (s+1)//2
        ans += (s + 1) // 2
        v += 1

    print(ans)

if __name__ == "__main__":
    main()