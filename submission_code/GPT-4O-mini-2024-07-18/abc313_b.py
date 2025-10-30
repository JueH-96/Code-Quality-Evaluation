def find_strongest_programmer(N, M, relations):
    # Create a list to count how many programmers each programmer is stronger than
    stronger_count = [0] * (N + 1)

    # Process the relations
    for A, B in relations:
        stronger_count[A] += 1  # A is stronger than B
        # B is not counted since we only count how many each programmer is stronger than

    # The strongest programmer should be stronger than N-1 others
    strongest_candidates = [i for i in range(1, N + 1) if stronger_count[i] == N - 1]

    # If there is exactly one strongest candidate, return it, otherwise return -1
    return strongest_candidates[0] if len(strongest_candidates) == 1 else -1

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
relations = [tuple(map(int, line.split())) for line in data[1:M + 1]]

# Get the result
result = find_strongest_programmer(N, M, relations)

# Print the result
print(result)