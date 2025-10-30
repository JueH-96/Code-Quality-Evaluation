import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))

    max_num = 200000
    spf = [0] * (max_num + 1)
    for i in range(2, max_num + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, max_num + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    Z = A.count(0)
    counts = defaultdict(int)
    for a in A:
        if a == 0:
            continue
        x = a
        res = 1
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                cnt += 1
                x //= p
            if cnt % 2 == 1:
                res *= p
        counts[res] += 1

    zero_pairs = (Z * (Z - 1)) // 2 + Z * (N - Z)
    non_zero_pairs = 0
    for c in counts.values():
        non_zero_pairs += c * (c - 1) // 2

    print(zero_pairs + non_zero_pairs)

if __name__ == "__main__":
    main()