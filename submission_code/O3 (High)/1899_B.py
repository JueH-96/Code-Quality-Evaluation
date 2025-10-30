import sys
import math

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)
    t = next(it)
    answers = []

    for _ in range(t):
        n = next(it)
        arr = [next(it) for _ in range(n)]

        # prefix sums
        pref = [0]*(n+1)
        for i, val in enumerate(arr, 1):
            pref[i] = pref[i-1] + val

        # collect all divisors of n
        divisors = []
        sq = int(math.isqrt(n))
        for d in range(1, sq+1):
            if n % d == 0:
                divisors.append(d)
                if d * d != n:
                    divisors.append(n // d)

        best = 0
        for k in divisors:          # k is the size of a truck load
            m = n // k              # number of trucks
            mn = float('inf')
            mx = -float('inf')

            # compute sums of consecutive blocks of length k
            for j in range(m):
                s = pref[(j+1)*k] - pref[j*k]
                if s < mn:
                    mn = s
                if s > mx:
                    mx = s
            diff = mx - mn
            if diff > best:
                best = diff

        answers.append(str(best))

    sys.stdout.write('
'.join(answers))

if __name__ == "__main__":
    main()