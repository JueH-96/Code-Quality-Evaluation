import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Function to calculate the XOR of a list of numbers
def xor_of_list(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Initialize the maximum XOR value
max_xor = 0

# Iterate through all combinations of K elements from A
for combo in combinations(A, K):
    current_xor = xor_of_list(combo)
    max_xor = max(max_xor, current_xor)

# Print the maximum XOR value
print(max_xor)