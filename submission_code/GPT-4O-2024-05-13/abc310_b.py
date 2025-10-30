# YOUR CODE HERE
def is_superior_product(N, M, products):
    for i in range(N):
        for j in range(N):
            if i != j:
                Pi, Ci, Fi = products[i]
                Pj, Cj, Fj = products[j]
                if Pi >= Pj and all(f in Fj for f in Fi):
                    if Pi > Pj or any(f not in Fi for f in Fj):
                        return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
products = []
index = 2
for _ in range(N):
    P = int(data[index])
    C = int(data[index + 1])
    F = list(map(int, data[index + 2: index + 2 + C]))
    products.append((P, C, F))
    index += 2 + C

print(is_superior_product(N, M, products))