# YOUR CODE HERE
import sys

# It is good practice to set a higher recursion limit for deep recursion,
# although the default limit (e.g., 1000) is usually sufficient for N <= 13.
sys.setrecursionlimit(2000)

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read problem parameters from standard input
        N = int(sys.stdin.readline())
        A = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Exit gracefully on input errors
        return

    # --- Global-like state for the recursive solver ---
    # Using a dictionary for memoization. Key: (start_index, level).
    memo = {}
    
    # Pre-calculating powers of 3 is a minor optimization.
    powers_of_3 = [1] * (N + 1)
    for i in range(1, N + 1):
        powers_of_3[i] = powers_of_3[i - 1] * 3

    # --- Recursive solver with memoization ---
    def solve(start, level):
        """
        Recursively computes the final value for a substring and the minimum cost to flip it.
        
        Args:
            start (int): The starting index of the substring in the original string A.
            level (int): The level of recursion, corresponding to a substring of length 3^level.

        Returns:
            A tuple (value, cost_to_flip):
            - value: The final single bit obtained from this subproblem.
            - cost_to_flip: The minimum number of flips in the leaves of this
                            subtree to change 'value'.
        """
        if (start, level) in memo:
            return memo[(start, level)]

        # Base case: a substring of length 1 (level 0).
        if level == 0:
            # The value is the bit itself. The cost to flip it is 1.
            return (int(A[start]), 1)

        # Recursive step:
        # Divide the problem into 3 subproblems of the next lower level.
        sub_len = powers_of_3[level - 1]
        
        v1, c1 = solve(start, level - 1)
        v2, c2 = solve(start + sub_len, level - 1)
        v3, c3 = solve(start + 2 * sub_len, level - 1)

        vals = [v1, v2, v3]
        costs = [c1, c2, c3]

        # Calculate the majority value and the cost to flip it.
        if sum(vals) >= 2:  # Majority is 1
            current_val = 1
            costs_of_1s = [costs[i] for i in range(3) if vals[i] == 1]
            if len(costs_of_1s) == 3:  # Case (1,1,1) -> flip two cheapest
                costs_of_1s.sort()
                cost_to_flip = costs_of_1s[0] + costs_of_1s[1]
            else:  # Case (1,1,0) -> flip cheapest. len(costs_of_1s) must be 2.
                cost_to_flip = min(costs_of_1s)
        else:  # Majority is 0
            current_val = 0
            costs_of_0s = [costs[i] for i in range(3) if vals[i] == 0]
            if len(costs_of_0s) == 3:  # Case (0,0,0) -> flip two cheapest
                costs_of_0s.sort()
                cost_to_flip = costs_of_0s[0] + costs_of_0s[1]
            else:  # Case (1,0,0) -> flip cheapest. len(costs_of_0s) must be 2.
                cost_to_flip = min(costs_of_0s)

        # Memoize the result and return.
        result = (current_val, cost_to_flip)
        memo[(start, level)] = result
        return result

    # --- Main execution ---
    # The final answer is the cost to flip the value of the entire string A.
    _, final_cost = solve(0, N)
    print(final_cost)

if __name__ == "__main__":
    main()