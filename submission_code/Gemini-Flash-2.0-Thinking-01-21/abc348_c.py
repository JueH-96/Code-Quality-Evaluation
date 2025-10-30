def solve():
    n = int(input())
    color_deliciousness_map = {}
    for _ in range(n):
        a, c = map(int, input().split())
        if c not in color_deliciousness_map:
            color_deliciousness_map[c] = []
        color_deliciousness_map[c].append(a)
    
    max_min_deliciousness = 0
    for color in color_deliciousness_map:
        deliciousness_values = color_deliciousness_map[color]
        min_deliciousness = min(deliciousness_values)
        max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)
        
    print(max_min_deliciousness)

if __name__ == '__main__':
    solve()