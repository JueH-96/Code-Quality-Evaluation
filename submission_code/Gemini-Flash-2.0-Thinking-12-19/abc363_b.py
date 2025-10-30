def solve():
    n, t, p = map(int, input().split())
    l = list(map(int, input().split()))
    
    days_needed = []
    for hair_length in l:
        days = max(0, t - hair_length)
        days_needed.append(days)
        
    days_needed.sort()
    
    result_days = days_needed[p-1]
    print(result_days)

if __name__ == '__main__':
    solve()