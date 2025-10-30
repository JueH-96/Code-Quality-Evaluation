def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    indices = {}
    for i in range(3 * n):
        if a[i] not in indices:
            indices[a[i]] = []
        indices[a[i]].append(i)
    
    f_values = {}
    for num in indices:
        f_values[num] = indices[num][1]
    
    sorted_nums = sorted(f_values.keys(), key=lambda x: f_values[x])
    
    print(*sorted_nums)

solve()