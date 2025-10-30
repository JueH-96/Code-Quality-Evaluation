def solve():
    n = int(input())
    a = list(map(int, input().split()))
    operations = []
    for _ in range(n - 1):
        s_i, t_i = map(int, input().split())
        operations.append({'s': s_i, 't': t_i})
    
    for i in range(n - 1):
        s_i = operations[i]['s']
        t_i = operations[i]['t']
        if a[i] >= s_i:
            k_i = a[i] // s_i
            a[i] -= k_i * s_i
            a[i+1] += k_i * t_i
            
    print(a[n-1])

if __name__ == '__main__':
    solve()