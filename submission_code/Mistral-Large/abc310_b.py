import sys
input = sys.stdin.read

def is_strictly_superior(N, M, products):
    for i in range(N):
        for j in range(N):
            if i != j:
                Pi, Ci, Fi = products[i]
                Pj, Cj, Fj = products[j]

                if Pi >= Pj and set(Fi).issubset(set(Fj)):
                    if Pi > Pj or len(Fi) < len(Fj):
                        return "Yes"
    return "No"

data = input().split()
N = int(data[0])
M = int(data[1])
products = []
index = 2

for _ in range(N):
    P = int(data[index])
    C = int(data[index + 1])
    F = [int(data[index + 2 + k]) for k in range(C)]
    products.append((P, C, F))
    index += 2 + C

print(is_strictly_superior(N, M, products))