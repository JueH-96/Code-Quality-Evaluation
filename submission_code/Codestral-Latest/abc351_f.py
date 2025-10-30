import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Initialize the result
result = 0

# Iterate over each element in the array
for i in range(N):
    for j in range(i+1, N):
        result += max(A[j] - A[i], 0)

print(result)