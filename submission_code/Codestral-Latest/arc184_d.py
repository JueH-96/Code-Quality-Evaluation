MOD = 998244353

def count_remaining_balls(N, balls):
    # Sort balls by their x-coordinates
    balls.sort()

    # Initialize a set to store the possible remaining balls
    remaining_sets = set()

    def dfs(index, current_set):
        if index == N:
            remaining_sets.add(tuple(sorted(current_set)))
            return

        # Include the current ball in the set
        dfs(index + 1, current_set | {index})

        # Exclude the current ball from the set
        dfs(index + 1, current_set)

    dfs(0, set())

    return len(remaining_sets) % MOD

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
balls = []
for i in range(N):
    x = int(data[2 * i + 1])
    y = int(data[2 * i + 2])
    balls.append((x, y))

# Calculate the result
result = count_remaining_balls(N, balls)

# Print the result
print(result)