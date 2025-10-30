import sys
from collections import deque

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1:]

    # Initialize a deque to keep track of the minimum cost for each string length
    min_cost = deque([(0, 0)])  # (cost, length)
    results = []

    for i in range(N):
        current_length = len(S[i])
        current_cost = current_length

        # Remove outdated entries from the deque
        while min_cost and min_cost[0][1] >= current_length:
            min_cost.popleft()

        # Calculate the cost to match any previous string
        if min_cost:
            best_cost = min_cost[0][0] + current_length
            current_cost = min(current_cost, best_cost)

        # Add the current string's cost to the deque
        while min_cost and min_cost[-1][0] >= current_cost:
            min_cost.pop()
        min_cost.append((current_cost, current_length))

        # Store the result for the current string
        results.append(current_cost)

    # Output the results
    for result in results:
        print(result)

if __name__ == "__main__":
    solve()