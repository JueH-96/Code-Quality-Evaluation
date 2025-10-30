def solve():
    n = int(input())
    beans = []
    for _ in range(n):
        a, c = map(int, input().split())
        beans.append((a, c))
    
    color_to_deliciousness = {}
    for a, c in beans:
        if c not in color_to_deliciousness:
            color_to_deliciousness[c] = []
        color_to_deliciousness[c].append(a)
    
    max_min_deliciousness = 0
    for color, deliciousness_values in color_to_deliciousness.items():
        min_deliciousness = min(deliciousness_values)
        max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)
    
    print(max_min_deliciousness)

solve()