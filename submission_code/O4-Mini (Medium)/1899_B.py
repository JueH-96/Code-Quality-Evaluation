import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    t = int(next(it))
    out = []
    import math
    for _ in range(t):
        n = int(next(it))
        a = [int(next(it)) for _ in range(n)]
        # build prefix sums
        pref = [0] * (n+1)
        for i in range(n):
            pref[i+1] = pref[i] + a[i]

        ans = 0
        # find all divisors of n
        divs = []
        r = int(math.isqrt(n))
        for d in range(1, r+1):
            if n % d == 0:
                divs.append(d)
                if d != n // d:
                    divs.append(n // d)

        # for each possible truck-size k
        for k in divs:
            num_trucks = n // k
            # if only one truck, diff = 0
            if num_trucks <= 1:
                continue
            # compute min and max segment-sum
            min_s = 10**30
            max_s = -10**30
            # segments are [0..k-1], [k..2k-1], ...
            for start in range(0, n, k):
                s = pref[start + k] - pref[start]
                if s < min_s:
                    min_s = s
                if s > max_s:
                    max_s = s
            diff = max_s - min_s
            if diff > ans:
                ans = diff

        out.append(str(ans))

    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()