import sys
sys.setrecursionlimit(10**7)

def main():
    import math
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = int(input_data[1])
    X = float(input_data[2])
    Y = float(input_data[3])

    # We memoize E(N): the minimum expected cost to reduce N to 0.
    # Base case: E(0) = 0
    #
    # Recurrence:
    #   E(N) = min( costA, costDice )
    # where
    #   costA = X + E(N // A),
    #
    #   costDice is derived from the fact that with probability 1/6 the
    #   chosen b = 1..6. If b=1, we remain at N (and pay E(N) again),
    #   else we go to floor(N/b). Solving the standard dice-expected-value
    #   equation leads to:
    #
    #   costDice = (6/5)*Y + (1/5)*[E(N//2) + E(N//3) + E(N//4) + E(N//5) + E(N//6)]
    #
    #   (that formula accounts for the possibility of b=1 keeping us
    #   in the same state, giving an infinite geometric series whose sum
    #   rearranges to the above closed form).
    #
    # Thus:
    #   E(N) = min(
    #       X + E(N//A),
    #       (6/5)*Y + (1/5)* sum_{b=2..6} E(N//b)
    #   )
    #
    # We implement this with a top-down memoized DFS. For large N (up to 1e18),
    # in practice, the recursion stays manageable because each call reduces N
    # significantly when dividing. Also many subproblems converge to the same
    # values of floor(N/b).

    from functools import lru_cache

    @lru_cache(None)
    def E(n):
        if n == 0:
            return 0.0
        # Cost if we choose operation A:
        costA = X + E(n // A)

        # Cost if we choose dice:
        # (6/5)*Y + (1/5)*[ E(n//2) + E(n//3) + E(n//4) + E(n//5) + E(n//6) ]
        s = 0.0
        for b in range(2, 7):
            s += E(n // b)
        costDice = (6.0/5.0)*Y + (1.0/5.0)*s

        return min(costA, costDice)

    ans = E(N)
    print(f"{ans:.9f}")

# Do not forget to call main at the end!
if __name__ == "__main__":
    main()