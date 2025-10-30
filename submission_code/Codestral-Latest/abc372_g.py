import sys
from math import gcd

input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1

def count_valid_pairs(N, A, B, C):
    min_x = float('inf')
    min_y = float('inf')

    for i in range(N):
        a, b, c = A[i], B[i], C[i]
        g = gcd(a, b)
        if c % g == 0:
            return 0
        min_x = min(min_x, (c - 1) // a)
        min_y = min(min_y, (c - 1) // b)

    return min_x * min_y

results = []
for _ in range(T):
    N = int(data[index])
    index += 1
    A = []
    B = []
    C = []
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index + 1]))
        C.append(int(data[index + 2]))
        index += 3
    results.append(count_valid_pairs(N, A, B, C))

for result in results:
    print(result)