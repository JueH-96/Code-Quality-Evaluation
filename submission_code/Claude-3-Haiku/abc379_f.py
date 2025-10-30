import sys

# Read the input
N, Q = map(int, input().split())
H = list(map(int, input().split()))
queries = [list(map(int, input().split())) for _ in range(Q)]

# Function to count the number of buildings that can be seen from both l and r
def count_visible_buildings(l, r):
    # Find the index of the rightmost building that can be seen from l
    left_idx = l - 1
    while left_idx >= 0 and H[left_idx] <= H[r - 1]:
        left_idx -= 1
    left_idx += 1

    # Find the index of the leftmost building that can be seen from r
    right_idx = r - 1
    while right_idx < N and H[right_idx] <= H[r - 1]:
        right_idx += 1
    right_idx -= 1

    # Count the number of buildings that can be seen from both l and r
    return max(0, right_idx - left_idx + 1)

# Solve the queries and print the results
for l, r in queries:
    print(count_visible_buildings(l, r))