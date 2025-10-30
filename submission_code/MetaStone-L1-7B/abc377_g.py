import sys
from collections import defaultdict

def main():
    MOD = 10**18 + 3
    BASE = 911382629

    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Preprocess all strings A to fill prefix_hash and suffix_hash
    prefix_hash = defaultdict(lambda: defaultdict(int))
    suffix_hash = defaultdict(lambda: defaultdict(int))

    for A in S[:N-1]:
        current_hash = 0
        for i in range(len(A)):
            current_hash = (current_hash * BASE + ord(A[i])) % MOD
            l = i + 1
            if prefix_hash[l][current_hash] > len(A):
                prefix_hash[l][current_hash] = len(A)

        current_hash = 0
        for i in range(len(A)-1, -1, -1):
            current_hash = (current_hash * BASE + ord(A[i])) % MOD
            l = i + 1
            if suffix_hash[l][current_hash] > len(A):
                suffix_hash[l][current_hash] = len(A)

    # For each T, compute minimal cost considering prefix and suffix
    for k in range(N):
        T = S[k]
        len_T = len(T)

        # Compute prefix-based cost
        min_cost_prefix = len_T  # delete all

        # Precompute the prefix hashes
        prefix_hashes = [0] * (len_T + 1)
        for i in range(len_T):
            prefix_hashes[i+1] = (prefix_hashes[i] * BASE + ord(T[i])) % MOD

        max_prefix_l = 0
        min_cost_prefix = len_T  # delete all
        for l in range(1, len_T + 1):
            hash_val = prefix_hashes[l]
            if hash_val in prefix_hash[l]:
                current_min_len = prefix_hash[l][hash_val]
                cost = len_T + current_min_len - 2 * l
                if cost < min_cost_prefix:
                    min_cost_prefix = cost
                    max_prefix_l = l

        # Compute suffix-based cost
        min_cost_suffix = len_T  # add all

        # Precompute the suffix hashes
        suffix_hashes = [0] * (len_T + 1)
        for i in range(len_T -1, -1, -1):
            suffix_hashes[i] = (suffix_hashes[i+1] * BASE + ord(T[i])) % MOD

        max_suffix_m = 0
        min_cost_suffix = len_T  # add all
        for m in range(1, len_T + 1):
            hash_val = suffix_hashes[m]
            if hash_val in suffix_hash[m]:
                current_min_len = suffix_hash[m][hash_val]
                cost = current_min_len + len_T - 2 * m
                if cost < min_cost_suffix:
                    min_cost_suffix = cost
                    max_suffix_m = m

        # The minimal cost is the minimum between prefix and suffix
        min_cost = min(min_cost_prefix, min_cost_suffix)

        print(min_cost)

if __name__ == '__main__':
    main()