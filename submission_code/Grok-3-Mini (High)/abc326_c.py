import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Parse N and M
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Parse the array A
A = [int(data[i]) for i in range(index, index + N)]

# Sort the array
A.sort()

# Sliding window to find the maximum number of gifts
left = 0
max_count = 0
for right in range(len(A)):
    # Move left pointer until the difference is less than M
    while A[right] - A[left] >= M:
        left += 1
    # Calculate the current window size and update max_count
    current_count = right - left + 1
    if current_count > max_count:
        max_count = current_count

# Output the result
print(max_count)