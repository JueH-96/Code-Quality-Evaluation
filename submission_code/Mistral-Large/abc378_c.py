import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Dictionary to store the most recent position of each element
last_position = {}

# List to store the result
B = []

for i in range(N):
    if A[i] in last_position:
        B.append(last_position[A[i]])
    else:
        B.append(-1)
    last_position[A[i]] = i

# Print the result
print(" ".join(map(str, B)))