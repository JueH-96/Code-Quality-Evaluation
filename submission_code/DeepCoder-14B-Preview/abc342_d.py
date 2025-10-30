import sys
from collections import defaultdict

def main():
    # Precompute smallest prime factors up to 200000
    max_a = 200000
    spf = list(range(max_a + 1))
    for i in range(2, int(max_a ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, max_a + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Read input
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))

    zero_count = 0
    square_free_counts = defaultdict(int)

    for x in a:
        if x == 0:
            zero_count += 1
        else:
            temp = x
            square_free = 1
            while temp != 1:
                p = spf[temp]
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    square_free *= p
            square_free_counts[square_free] += 1

    # Calculate zero_pairs
    zero_pairs = zero_count * (n - zero_count) + (zero_count * (zero_count - 1)) // 2

    # Calculate non-zero pairs
    non_zero_pairs = 0
    for count in square_free_counts.values():
        non_zero_pairs += count * (count - 1) // 2

    total = zero_pairs + non_zero_pairs
    print(total)

if __name__ == "__main__":
    main()