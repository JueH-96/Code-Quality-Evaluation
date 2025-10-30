def solve():
    n = int(input())
    a = list(map(int, input().split()))
    operations = []
    for _ in range(n - 1):
        s_t = list(map(int, input().split()))
        operations.append({'s': s_t[0], 't': s_t[1]})
    
    currencies = list(a)
    
    for i in range(n - 1):
        s_i = operations[i]['s']
        t_i = operations[i]['t']
        if currencies[i] >= s_i:
            times = currencies[i] // s_i
            currencies[i] -= times * s_i
            currencies[i+1] += times * t_i
            
    print(currencies[n-1])

if __name__ == '__main__':
    solve()