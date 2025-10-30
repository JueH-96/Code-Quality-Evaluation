# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

products = []
index = 2
for i in range(N):
    P = int(data[index])
    C = int(data[index + 1])
    F = list(map(int, data[index + 2:index + 2 + C]))
    products.append((P, C, F))
    index += 2 + C

for i in range(N):
    for j in range(N):
        if i != j:
            P_i, C_i, F_i = products[i]
            P_j, C_j, F_j = products[j]
            if P_i >= P_j and all(f in F_j for f in F_i):
                if P_i > P_j or any(f not in F_i for f in F_j):
                    print("Yes")
                    exit()

print("No")