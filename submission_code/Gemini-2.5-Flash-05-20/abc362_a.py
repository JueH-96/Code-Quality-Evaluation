# YOUR CODE HERE
import sys

def solve():
    # Read the first line for R, G, B
    r_str, g_str, b_str = sys.stdin.readline().split()
    R = int(r_str)
    G = int(g_str)
    B = int(b_str)

    # Read the second line for C
    C = sys.stdin.readline().strip()

    # Create a list to store the costs of pens Takahashi can buy
    available_costs = []

    # Check if Takahashi can buy a red pen
    if C != "Red":
        available_costs.append(R)

    # Check if Takahashi can buy a green pen
    if C != "Green":
        available_costs.append(G)

    # Check if Takahashi can buy a blue pen
    if C != "Blue":
        available_costs.append(B)

    # Find the minimum cost among the available pens
    # The problem constraints ensure that at least two colors are available,
    # so available_costs will never be empty.
    min_cost = min(available_costs)

    # Print the result
    print(min_cost)

# Call the solve function to execute the program
solve()