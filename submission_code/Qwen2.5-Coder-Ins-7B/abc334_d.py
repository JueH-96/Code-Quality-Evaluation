import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
R = list(map(int, data[2:2+N]))
queries = list(map(int, data[2+N:]))

# Sort the reindeer requirements in descending order
R.sort(reverse=True)

# Calculate the prefix sum of the sorted reindeer requirements
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i+1] = prefix_sum[i] + R[i]

# Function to find the maximum number of sleighs that can be pulled with X reindeer
def max_sleighs(X):
    left, right = 0, N
    while left < right:
        mid = (left + right + 1) // 2
        if prefix_sum[mid] <= X:
            left = mid
        else:
            right = mid - 1
    return left

# Process each query
results = [max_sleighs(X) for X in queries]

# Print the results
for result in results:
    print(result)