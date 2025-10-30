from typing import List

def min_sensor_cost(N: int, D: List[int], L: List[int], C: List[int], K: List[int]) -> int:
    # Initialize the minimum cost to infinity
    min_cost = float('inf')

    # Try all possible combinations of sensors
    for use_type1 in range(K[0] + 1):
        for use_type2 in range(K[1] + 1):
            # Calculate the total cost of the current combination
            cost = use_type1 * C[0] + use_type2 * C[1]

            # Check if all sections can be monitored
            covered = [False] * N
            for i in range(N):
                # Try to cover the current section with the available sensors
                for j in range(use_type1):
                    if D[i] <= L[0]:
                        covered[i] = True
                        break
                    D[i] -= L[0]
                for j in range(use_type2):
                    if D[i] <= L[1]:
                        covered[i] = True
                        break
                    D[i] -= L[1]

            # If all sections are covered, update the minimum cost
            if all(covered):
                min_cost = min(min_cost, cost)

    # If it's impossible to monitor all sections, return -1
    return min_cost if min_cost != float('inf') else -1

# Read the input
N = int(input())
D = [int(x) for x in input().split()]
L = [int(x) for x in input().split()]
C = [int(x) for x in input().split()]
K = [int(x) for x in input().split()]

# Print the result
print(min_sensor_cost(N, D, L, C, K))