from itertools import combinations

def count_valid_divisions(N, T, M, incompatible_pairs):
    # Generate all possible ways to divide N players into T teams
    def generate_divisions(N, T):
        for combo in combinations(range(N), T - 1):
            division = [0] * N
            for i, boundary in enumerate(combo):
                for j in range(boundary + 1):
                    division[j] = i
            for j in range(combo[-1] + 1, N):
                division[j] = T - 1
            yield division

    valid_count = 0

    for division in generate_divisions(N, T):
        valid = True
        for a, b in incompatible_pairs:
            if division[a - 1] == division[b - 1]:
                valid = False
                break
        if valid:
            valid_count += 1

    return valid_count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
M = int(data[2])

incompatible_pairs = []
for i in range(M):
    A_i = int(data[3 + 2 * i])
    B_i = int(data[4 + 2 * i])
    incompatible_pairs.append((A_i, B_i))

# Calculate and print the result
result = count_valid_divisions(N, T, M, incompatible_pairs)
print(result)