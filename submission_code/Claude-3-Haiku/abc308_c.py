import sys

# Read the input
N = int(sys.stdin.readline().strip())
people = []
for _ in range(N):
    A, B = map(int, sys.stdin.readline().strip().split())
    people.append((i+1, A, B, A/(A+B)))

# Sort the people in descending order of their success rates, with ties broken in ascending order of their assigned numbers
people.sort(key=lambda x: (-x[3], x[0]))

# Print the sorted people
print(" ".join(map(str, [p[0] for p in people])))