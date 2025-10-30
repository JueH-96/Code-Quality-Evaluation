import sys

def solve():
    N, X = map(int, input().split())
    machines = []
    for _ in range(N):
        A, P, B, Q = map(int, input().split())
        machines.append((A, P, B, Q))
    
    max_capacity = 0
    for mask in range(1 << N):
        cost = 0
        capacity = float('inf')
        for i in range(N):
            if mask & (1 << i):
                A, P, B, Q = machines[i]
                cost += P
                capacity = min(capacity, A)
            else:
                A, P, B, Q = machines[i]
                cost += Q
                capacity = min(capacity, B)
        if cost <= X:
            max_capacity = max(max_capacity, capacity)
    
    print(max_capacity)

solve()