# YOUR CODE HERE
def find_pairs(N, M, A):
    # Calculate the prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + A[i - 1]
    
    # Calculate the remainders of prefix sums modulo M
    remainders = [prefix_sums[i] % M for i in range(N + 1)]
    
    # Count the occurrences of each remainder
    remainder_count = {}
    for rem in remainders:
        if rem in remainder_count:
            remainder_count[rem] += 1
        else:
            remainder_count[rem] = 1
    
    # Calculate the number of valid pairs
    count = 0
    for rem in remainder_count:
        if remainder_count[rem] > 1:
            count += (remainder_count[rem] * (remainder_count[rem] - 1)) // 2
    
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

print(find_pairs(N, M, A))