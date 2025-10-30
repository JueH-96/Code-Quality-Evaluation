def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        a, b, c = map(int, input().split())
        edges.append((a, b, c))

    total_cost = 0
    for _, _, c in edges:
        total_cost += c
    
    print(2 * total_cost - max(c for _,_,c in edges))

solve()