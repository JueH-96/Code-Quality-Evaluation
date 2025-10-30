def solve():
    n, m = map(int, input().split())
    a_prices = list(map(int, input().split()))
    b_prices = list(map(int, input().split()))
    
    low = 1
    high = 10**9 + 1
    min_x = -1
    
    while low < high:
        mid = (low + high) // 2
        sellers_count = 0
        for price in a_prices:
            if price <= mid:
                sellers_count += 1
        buyers_count = 0
        for price in b_prices:
            if price >= mid:
                buyers_count += 1
                
        if sellers_count >= buyers_count:
            high = mid
        else:
            low = mid + 1
            
    print(low)

if __name__ == '__main__':
    solve()