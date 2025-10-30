# Read the input
N = int(input())
H = list(map(int, input().split()))

# Initialize the result list
result = [0] * N

# Iterate through the buildings from right to left
max_height = 0
for i in range(N-1, -1, -1):
    if H[i] > max_height:
        max_height = H[i]
    else:
        result[i] = N - i - 1

# Print the result
print(" ".join(map(str, result)))