import sys
from collections import Counter

def main() -> None:
    sys.setrecursionlimit(1000000)

    # read input
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # frequency of every character that actually appears
    freq = Counter(S)
    letters = sorted(freq.keys())                 # fixed order for indices
    init_counts = tuple(freq[c] for c in letters) # tuple of remaining amounts

    K_minus_1 = K - 1                             # used many times
    memo = {}                                     # DP memoisation

    # ------------------------------------------------------------------ #
    #  dfs(counts, suffix) returns the number of ways to finish the word #
    #  • counts : tuple with remaining quantities of each letter         #
    #  • suffix : last (≤ K-1) letters of the already built prefix       #
    # ------------------------------------------------------------------ #
    def dfs(counts: tuple, suffix: str) -> int:
        key = (counts, suffix)
        if key in memo:
            return memo[key]

        if sum(counts) == 0:          # whole string is finished
            memo[key] = 1
            return 1

        total = 0
        for idx, ch in enumerate(letters):
            if counts[idx] == 0:
                continue

            new_suffix = suffix + ch

            # check the only new substring of length K (the one that ends here)
            if len(new_suffix) >= K:
                last_k = new_suffix[-K:]
                if last_k == last_k[::-1]:      # it is a palindrome → invalid
                    continue

            # keep only the last K-1 characters in the suffix
            if K_minus_1 > 0 and len(new_suffix) > K_minus_1:
                new_suffix = new_suffix[-K_minus_1:]

            # recurse with one fewer occurrence of this character
            nxt_counts = list(counts)
            nxt_counts[idx] -= 1
            total += dfs(tuple(nxt_counts), new_suffix)

        memo[key] = total
        return total

    print(dfs(init_counts, ""))

if __name__ == "__main__":
    main()