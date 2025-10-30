import sys

def min_cost_to_make_good_string(n, s, costs):
    """
    Calculate the minimum cost to make the string S a good string.

    Args:
    n (int): The length of the string S.
    s (str): The string S consisting of 0 and 1.
    costs (list): A list of costs where costs[i] is the cost of flipping the i-th character of S.

    Returns:
    int: The minimum total cost required to make S a good string.
    """
    # Initialize the minimum cost to infinity
    min_cost = float('inf')

    # Iterate over all possible positions where the two consecutive characters are the same
    for i in range(n - 1):
        # Calculate the cost of making the i-th and (i + 1)-th characters the same
        cost = 0
        if s[i] != s[i + 1]:
            cost += min(costs[i], costs[i + 1])

        # Calculate the cost of making all other consecutive characters different
        for j in range(n - 1):
            if j != i:
                if s[j] == s[j + 1]:
                    cost += min(costs[j], costs[j + 1])

        # Update the minimum cost
        min_cost = min(min_cost, cost)

    return min_cost

def main():
    # Read the input from stdin
    n = int(input())
    s = input()
    costs = list(map(int, input().split()))

    # Calculate the minimum cost to make S a good string
    min_cost = min_cost_to_make_good_string(n, s, costs)

    # Print the answer to stdout
    print(min_cost)

if __name__ == "__main__":
    main()