import sys

def solve():
    N, M = map(int, input().split())
    products = []
    for _ in range(N):
        P, C, *F = map(int, input().split())
        functions = set(F)
        products.append((P, functions))
    
    for i in range(N):
        for j in range(N):
            if i != j and products[i][0] >= products[j][0] and products[j][1].issuperset(products[i][1]):
                if products[i][0] > products[j][0] or len(products[j][1] - products[i][1]) > 0:
                    print("Yes")
                    return
    print("No")

solve()