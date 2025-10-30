import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    prices = list(map(int, sys.stdin.readline().split()))
    l_values = list(map(int, sys.stdin.readline().split()))
    d_values = list(map(int, sys.stdin.readline().split()))
    
    items_with_indices = []
    for i in range(n):
        items_with_indices.append({'price': prices[i], 'index': i})
    items_with_indices.sort(key=lambda x: x['price'], reverse=True)
    
    coupons = []
    for i in range(m):
        coupons.append({'l': l_values[i], 'd': d_values[i], 'index': i})
    coupons.sort(key=lambda x: x['d'], reverse=True)
    
    coupon_used = [False] * m
    item_couponed = [False] * n
    total_cost = 0
    
    for item_info in items_with_indices:
        item_price = item_info['price']
        item_index = item_info['index']
        best_coupon_index = -1
        best_discount = 0
        
        for i in range(m):
            coupon = coupons[i]
            if not coupon_used[coupon['index']] and item_price >= coupon['l']:
                if coupon['d'] > best_discount:
                    best_discount = coupon['d']
                    best_coupon_index = i
                    
        if best_coupon_index != -1:
            coupon = coupons[best_coupon_index]
            coupon_used[coupon['index']] = True
            item_couponed[item_index] = True
            total_cost += item_price - coupon['d']
        else:
            total_cost += item_price
            
    print(total_cost)

if __name__ == '__main__':
    solve()