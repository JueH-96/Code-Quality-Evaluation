def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    f_values = {}
    for i in range(1, n + 1):
        indices = [j for j, x in enumerate(a) if x == i]
        f_values[i] = indices[1]
    
    sorted_items = sorted(f_values.items(), key=lambda item: item[1])
    
    result = [item[0] for item in sorted_items]
    print(*result)

solve()