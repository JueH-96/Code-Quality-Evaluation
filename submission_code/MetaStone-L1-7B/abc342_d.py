import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))

    max_A = 200000  # Since A_i can be up to 2e5

    # Precompute smallest prime factors (SPF) up to max_A
    spf = list(range(max_A + 1))
    for i in range(2, int(max_A**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_A + 1, i):
                if spf[j] == j:
                    spf[j] = i

    count = defaultdict(int)
    count_zero = 0

    for x in A:
        if x == 0:
            s = 0
        else:
            temp = x
            factors = set()
            while temp != 1:
                p = spf[temp]
                cnt = 0
                while temp % p == 0:
                    temp = temp // p
                    cnt += 1
                if cnt % 2 == 1:
                    factors.add(p)
            s = 1
            for p in factors:
                s *= p
        count[s] += 1
        if s == 0:
            count_zero += 1

    sum_pairs = 0
    for k in count.values():
        sum_pairs += k * (k - 1) // 2

    ans = sum_pairs + count_zero * (N - count_zero)
    print(ans)

if __name__ == '__main__':
    main()