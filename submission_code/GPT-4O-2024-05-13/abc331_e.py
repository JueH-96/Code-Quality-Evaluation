# YOUR CODE HERE
def find_most_expensive_set_meal(N, M, L, a, b, forbidden_pairs):
    forbidden_set = set((c-1, d-1) for c, d in forbidden_pairs)
    max_price = -1
    
    for i in range(N):
        for j in range(M):
            if (i, j) not in forbidden_set:
                max_price = max(max_price, a[i] + b[j])
    
    return max_price

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
L = int(data[2])

a = list(map(int, data[3:3+N]))
b = list(map(int, data[3+N:3+N+M]))

forbidden_pairs = []
index = 3 + N + M
for _ in range(L):
    c = int(data[index])
    d = int(data[index+1])
    forbidden_pairs.append((c, d))
    index += 2

print(find_most_expensive_set_meal(N, M, L, a, b, forbidden_pairs))