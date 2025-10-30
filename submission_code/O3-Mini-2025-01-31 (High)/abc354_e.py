import sys
from functools import lru_cache

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = [0] * n
    B = [0] * n
    idx = 1
    for i in range(n):
        A[i] = int(data[idx])
        B[i] = int(data[idx + 1])
        idx += 2

    # Precompute all candidate pairs that can be removed
    # A move is allowed if either the front sides or the back sides match.
    candidate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i] == A[j] or B[i] == B[j]:
                candidate_pairs.append((i, j))
    
    # Use memoization with dp(mask) where mask is an integer
    # representing the subset of remaining cards. Bit i is 1 if card i remains.
    @lru_cache(maxsize=None)
    def dp(mask):
        # For each candidate move, if both cards are present in the current mask,
        # try removing them. If in any move the opponent is forced to a losing position,
        # then the current player wins.
        for i, j in candidate_pairs:
            if (mask >> i) & 1 and (mask >> j) & 1:
                new_mask = mask & ~((1 << i) | (1 << j))
                if not dp(new_mask):
                    return True
        return False

    initial_mask = (1 << n) - 1  # All cards are available initially.
    if dp(initial_mask):
        sys.stdout.write("Takahashi")
    else:
        sys.stdout.write("Aoki")

if __name__ == '__main__':
    main()