def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    c = list(map(int, input().split()))
    
    values = []
    for i in range(n):
        for j in range(n):
            for l in range(n):
                values.append(a[i] * b[j] + b[j] * c[l] + c[l] * a[i])
    
    values.sort(reverse=True)
    print(values[k-1])

solve()