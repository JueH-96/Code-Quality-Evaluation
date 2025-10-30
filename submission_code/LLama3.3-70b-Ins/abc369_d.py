import sys

def max_experience_points(N, A):
    """
    Calculate the maximum total experience points that can be gained from N monsters.

    Parameters:
    N (int): The number of monsters.
    A (list): A list of monster strengths.

    Returns:
    int: The maximum total experience points.
    """
    # Initialize a list to store the maximum experience points for each subproblem
    dp = [0] * (N + 1)

    # Iterate over each monster
    for i in range(1, N + 1):
        # Initialize the maximum experience points for the current monster to 0
        max_exp = 0

        # Iterate over each possible number of defeated monsters
        for j in range(i + 1):
            # Calculate the experience points for the current number of defeated monsters
            exp = sum(A[k - 1] for k in range(1, j + 1)) + sum(A[k - 1] for k in range(2, j + 1, 2))

            # Update the maximum experience points for the current monster
            max_exp = max(max_exp, exp)

        # Update the maximum experience points for the current subproblem
        dp[i] = max(dp[i - 1], max_exp)

    # Return the maximum total experience points
    return dp[N]

# Read the input from stdin
N = int(input())
A = list(map(int, input().split()))

# Calculate and print the maximum total experience points
print(max_experience_points(N, A))