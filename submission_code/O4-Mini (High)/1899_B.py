import sys

def main():
    input = sys.stdin.readline
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        # build prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]

        ans = 0
        # enumerate all divisors k of n
        i = 1
        while i * i <= n:
            if n % i == 0:
                # first divisor k = i
                k = i
                max_s = prefix[k]
                min_s = max_s
                j = k
                while j <= n:
                    s = prefix[j] - prefix[j - k]
                    if s > max_s: max_s = s
                    if s < min_s: min_s = s
                    j += k
                diff = max_s - min_s
                if diff > ans:
                    ans = diff

                # second divisor k = n // i (if different)
                if i != n // i:
                    k = n // i
                    max_s = prefix[k]
                    min_s = max_s
                    j = k
                    while j <= n:
                        s = prefix[j] - prefix[j - k]
                        if s > max_s: max_s = s
                        if s < min_s: min_s = s
                        j += k
                    diff = max_s - min_s
                    if diff > ans:
                        ans = diff
            i += 1

        print(ans)

main()