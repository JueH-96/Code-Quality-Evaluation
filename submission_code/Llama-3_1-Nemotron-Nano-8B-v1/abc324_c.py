import sys
from bisect import bisect_left
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T_prime = input[ptr]
    ptr += 1
    S_list = []
    for _ in range(N):
        S_list.append(input[ptr])
        ptr += 1

    # Precompute character indices for T_prime
    t_char_indices = defaultdict(list)
    for idx, c in enumerate(T_prime):
        t_char_indices[c].append(idx)

    result = []
    len_T_prime = len(T_prime)

    for s_idx, S in enumerate(S_list):
        len_S = len(S)
        # Check condition1: S == T_prime
        if S == T_prime:
            result.append(s_idx + 1)
            continue
        # Check condition4: len_S == len_T_prime and exactly one differing character
        if len_S == len_T_prime:
            diff = 0
            for sc, tc in zip(S, T_prime):
                if sc != tc:
                    diff += 1
                if diff > 1:
                    break
            if diff == 1:
                result.append(s_idx + 1)
                continue
        # Check condition2: len_S == len_T_prime -1 and S is a subsequence of T_prime
        if len_S == len_T_prime - 1:
            it = 0
            valid = True
            for c in S:
                if c not in t_char_indices:
                    valid = False
                    break
                idx_list = t_char_indices[c]
                pos = bisect_left(idx_list, it)
                if pos >= len(idx_list):
                    valid = False
                    break
                it = idx_list[pos] + 1
            if valid:
                result.append(s_idx + 1)
                continue
        # Check condition3: len_S == len_T_prime +1 and T_prime is a subsequence of S
        if len_S == len_T_prime + 1:
            it = 0
            valid = True
            for c in T_prime:
                found = False
                while it < len(S):
                    if S[it] == c:
                        found = True
                        it += 1
                        break
                    it += 1
                if not found:
                    valid = False
                    break
            if valid:
                result.append(s_idx + 1)
                continue

    # Output the result
    result.sort()
    print(len(result))
    if result:
        print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()