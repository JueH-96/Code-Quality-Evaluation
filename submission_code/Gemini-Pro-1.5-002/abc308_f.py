# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    p = list(map(int, input().split()))
    l = list(map(int, input().split()))
    d = list(map(int, input().split()))

    p.sort()
    coupons = []
    for i in range(m):
        coupons.append((l[i], d[i]))
    
    coupons.sort(key=lambda x: x[1], reverse=True)

    total_cost = 0
    used_coupons = [False] * m
    
    for price in p:
        best_coupon = -1
        for i in range(m):
            if not used_coupons[i] and price >= coupons[i][0]:
                if best_coupon == -1 or coupons[i][1] > coupons[best_coupon][1]:
                    best_coupon = i
        
        if best_coupon != -1:
            total_cost += price - coupons[best_coupon][1]
            used_coupons[best_coupon] = True
        else:
            total_cost += price
            
    print(total_cost)

solve()