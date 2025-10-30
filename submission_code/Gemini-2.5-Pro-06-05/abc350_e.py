import sys

def main():
    """
    Reads input, solves the problem using recursion with memoization,
    and prints the result.
    """
    # Set a higher recursion limit for safety, although not strictly necessary for this problem
    # as the maximum recursion depth is well within Python's default limit.
    sys.setrecursionlimit(2 * 10**5)
    
    # Read problem parameters from standard input
    N, A, X, Y = map(int, sys.stdin.readline().split())

    # Dictionary for memoization to store results of f(n)
    memo = {}

    def f(n):
        """
        Recursively calculates the minimum expected cost to reduce n to 0.
        
        Args:
            n: The current integer value.
        
        Returns:
            The minimum expected cost to reach 0 from n.
        """
        # Base case: if n is 0, cost is 0.
        if n == 0:
            return 0.0
        
        # If the result for n is already computed, return it.
        if n in memo:
            return memo[n]

        # Option A: Pay X, divide by A.
        cost_a = float(X) + f(n // A)

        # Option B: Pay Y, roll a die.
        # The expected cost is derived by solving the recurrence E(n) = Y + (1/6)E(n) + ...
        sum_f_b = 0.0
        for b in range(2, 7):
            sum_f_b += f(n // b)
        cost_b = (float(Y) * 6.0) / 5.0 + sum_f_b / 5.0
        
        # The optimal choice is the minimum of the two costs.
        result = min(cost_a, cost_b)
        
        # Memoize the result before returning.
        memo[n] = result
        return result

    # Calculate the final answer for the initial N.
    answer = f(N)
    
    # Print the answer with high precision.
    print(f"{answer:.15f}")

if __name__ == "__main__":
    main()