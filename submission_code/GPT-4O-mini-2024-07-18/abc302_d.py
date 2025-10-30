def max_gift_sum(N, M, D, A, B):
    A.sort()
    B.sort()
    
    max_sum = -1
    j = 0
    
    for a in A:
        while j < M and B[j] < a - D:
            j += 1
        if j < M and B[j] <= a + D:
            max_sum = max(max_sum, a + B[j])
        if j > 0 and B[j - 1] <= a + D:
            max_sum = max(max_sum, a + B[j - 1])
    
    return max_sum

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
D = int(data[2])
A = list(map(int, data[3:3 + N]))
B = list(map(int, data[3 + N:3 + N + M]))

result = max_gift_sum(N, M, D, A, B)
print(result)