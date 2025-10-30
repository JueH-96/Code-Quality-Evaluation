import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Sort the list in descending order to maximize the experience points
A.sort(reverse=True)

# Calculate the maximum experience points
total_experience = 0
for i in range(N):
    if (i + 1) % 2 == 0:
        total_experience += 2 * A[i]
    else:
        total_experience += A[i]

print(total_experience)