import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Initialize DP: cnt[m] is a dictionary of xor -> count for size mod m
    cnt = [defaultdict(int) for _ in range(M)]
    cnt[0][0] = 1

    for a in A:
        new_cnt = [defaultdict(int) for _ in range(M)]
        for m in range(M):
            # Process not taking a
            for x, c in cnt[m].items():
                new_cnt[m][x] = (new_cnt[m][x] + c) % MOD
            # Process taking a
            new_m = (m + 1) % M
            for x, c in cnt[m].items():
                new_x = x ^ a
                new_cnt[new_m][new_x] = (new_cnt[new_m][new_x] + c) % MOD
        cnt = new_cnt

    # Precompute x^K mod MOD for all possible x up to 2^20-1
    max_x = (1 << 20) - 1
    powers = [0] * (max_x + 1)
    for x in range(max_x + 1):
        powers[x] = pow(x, K, MOD)

    # Calculate the answer
    ans = 0
    total = 0
    empty_count = cnt[0].get(0, 0)
    for x in cnt[0]:
        c = cnt[0][x]
        if x == 0:
            # Subtract the empty subset (counted once)
            c = (c - 1) % MOD
        total = (total + c * powers[x]) % MOD
    ans = total % MOD
    print(ans)

if __name__ == '__main__':
    main()