def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    def count_sellers(price):
        count = 0
        for seller_price in a:
            if seller_price <= price:
                count += 1
        return count
        
    def count_buyers(price):
        count = 0
        for buyer_price in b:
            if buyer_price >= price:
                count += 1
        return count
        
    low = 1
    high = 10**9 + 1
    min_price = -1
    
    while low <= high:
        mid_price = (low + high) // 2
        sellers_count = count_sellers(mid_price)
        buyers_count = count_buyers(mid_price)
        
        if sellers_count >= buyers_count:
            min_price = mid_price
            high = mid_price - 1
        else:
            low = mid_price + 1
            
    print(min_price)

if __name__ == '__main__':
    solve()