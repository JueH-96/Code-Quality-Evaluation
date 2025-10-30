def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    f_values = {}
    for i in range(1, n + 1):
        indices = []
        for j in range(3 * n):
            if a[j] == i:
                indices.append(j + 1)
        f_values[i] = indices[1]
    
    sorted_numbers = sorted(range(1, n + 1), key=lambda x: f_values[x])
    print(*sorted_numbers)

solve()