# YOUR CODE HERE
def count_arithmetic_subsequences(N, A):
    if N == 1:
        return 1
    
    count = 0
    length = 1
    
    for i in range(1, N):
        if i == 1 or A[i] - A[i-1] == A[i-1] - A[i-2]:
            length += 1
        else:
            count += length * (length + 1) // 2
            length = 1
    
    count += length * (length + 1) // 2
    
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(count_arithmetic_subsequences(N, A))