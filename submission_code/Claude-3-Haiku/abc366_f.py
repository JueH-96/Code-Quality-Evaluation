import sys

def solve():
    N, K = map(int, input().split())
    functions = []
    for _ in range(N):
        A, B = map(int, input().split())
        functions.append((A, B))
    
    max_value = float('-inf')
    for p in range(1, N+1):
        for q in range(1, N+1):
            if p != q:
                for r in range(1, N+1):
                    if r != p and r != q:
                        value = functions[p-1][0] * (functions[q-1][0] * 1 + functions[q-1][1]) + functions[p-1][1]
                        value = functions[r-1][0] * value + functions[r-1][1]
                        max_value = max(max_value, value)
    
    print(int(max_value))

solve()