import sys
import threading

def main():
    import math
    N = int(sys.stdin.readline().strip())
    # Collect all perfect powers a^b with b >= 3, a >= 2
    seen = set()
    # maximum exponent b needed: 2^60 is ~1e18, so b up to 60
    for b in range(3, 61):
        a = 2
        while True:
            p = pow(a, b)
            if p > N:
                break
            seen.add(p)
            a += 1

    # Count how many squares > 1 up to N:
    sqrtN = math.isqrt(N)
    # squares from 2^2 up to sqrtN^2
    squares_count = max(0, sqrtN - 1)

    # Among seen, some are also perfect squares; we must not double-count them.
    overlap = 0
    for x in seen:
        # check if x is a perfect square >= 2^2
        r = math.isqrt(x)
        if r >= 2 and r * r == x:
            overlap += 1

    # Unique perfect powers > 1:
    unique_gt1 = len(seen) + (squares_count - overlap)

    # 1 = 1^b also counts if N >= 1
    ans = unique_gt1 + (1 if N >= 1 else 0)
    print(ans)

if __name__ == "__main__":
    main()