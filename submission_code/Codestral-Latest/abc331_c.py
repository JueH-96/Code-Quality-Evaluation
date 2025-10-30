# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Create a dictionary to count the occurrences of each element
count = defaultdict(int)
for num in A:
    count[num] += 1

# Create a list of unique elements sorted in descending order
unique_sorted = sorted(count.keys(), reverse=True)

# Calculate the prefix sum of the counts
prefix_sum = [0] * (len(unique_sorted) + 1)
for i in range(len(unique_sorted)):
    prefix_sum[i + 1] = prefix_sum[i] + count[unique_sorted[i]] * unique_sorted[i]

# Calculate the result for each A_i
result = []
for num in A:
    idx = unique_sorted.index(num)
    result.append(prefix_sum[idx])

print(" ".join(map(str, result)))