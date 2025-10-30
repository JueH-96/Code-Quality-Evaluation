# YOUR CODE HERE
def count_kagamimochi(N, A):
    count = 0
    j = 0
    for i in range(1, N):
        while j < i and A[j] * 2 < A[i]:
            j += 1
        count += i - j
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(count_kagamimochi(N, A))