def solve():
    n, m, p = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    b_sorted = sorted(b)
    prefix_sum_b = [0] * (m + 1)
    for i in range(m):
        prefix_sum_b[i+1] = prefix_sum_b[i] + b_sorted[i]
        
    total_price = 0
    for i in range(n):
        target_value = p - a[i]
        k_index = 0
        low = 0
        high = m - 1
        best_index = -1
        while low <= high:
            mid_index = (low + high) // 2
            if b_sorted[mid_index] <= target_value:
                best_index = mid_index
                low = mid_index + 1
            else:
                high = mid_index - 1
                
        k_i = 0
        if best_index != -1:
            k_i = best_index + 1
            
        price_i = k_i * a[i] + prefix_sum_b[k_i] + (m - k_i) * p
        total_price += price_i
        
    print(total_price)

if __name__ == '__main__':
    solve()