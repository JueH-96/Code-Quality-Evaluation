def solve():
    n, t, p = map(int, input().split())
    l = list(map(int, input().split()))
    
    initial_count = 0
    for hair_length in l:
        if hair_length >= t:
            initial_count += 1
            
    if initial_count >= p:
        print(0)
    else:
        required_days = []
        for hair_length in l:
            required_days.append(max(0, t - hair_length))
        
        required_days.sort()
        print(required_days[p-1])

if __name__ == '__main__':
    solve()