# YOUR CODE HERE
def longest_common_prefix(x, y):
    min_len = min(len(x), len(y))
    for i in range(min_len):
        if x[i] != y[i]:
            return i
    return min_len

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:]

total_sum = 0

for i in range(N-1):
    for j in range(i+1, N):
        total_sum += longest_common_prefix(strings[i], strings[j])

print(total_sum)