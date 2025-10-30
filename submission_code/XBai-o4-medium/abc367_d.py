import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N

    prefix_mod = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_mod[i] = (prefix_mod[i-1] + A[i-1]) % M

    total_sum_mod = prefix_mod[N]

    # Compute case1: t > s
    freq1 = defaultdict(int)
    case1 = 0
    for t in range(1, N + 1):
        current_mod = prefix_mod[t-1]
        case1 += freq1[current_mod]
        freq1[current_mod] += 1

    # Compute case2: t < s
    freq2 = defaultdict(int)
    case2 = 0
    for s in range(1, N + 1):
        current_prefix = prefix_mod[s-1]
        target = (current_prefix - total_sum_mod) % M
        case2 += freq2[target]
        freq2[current_prefix] += 1

    print(case1 + case2)

if __name__ == '__main__':
    main()