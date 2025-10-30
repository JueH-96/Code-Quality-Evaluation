def solve():
    N, M, P = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    B.sort()  # Sort the side dishes by their prices
    prefix_sum_B = [0]  # Prefix sums for B
    for b in B:
        prefix_sum_B.append(prefix_sum_B[-1] + b)
    
    total_price = 0
    for a in A:
        # Find the smallest j such that a + B[j] > P, using binary search
        left, right = 0, M
        while left < right:
            mid = (left + right) // 2
            if a + B[mid] > P:
                right = mid
            else:
                left = mid + 1
        
        j_first = left  # Smallest j such that a + B[j] > P (might be M if none exist)
        
        # For the first j_first side dishes, the price is a + B[j]
        if j_first > 0:
            sum_first = a * j_first + prefix_sum_B[j_first]
            total_price += sum_first
        
        # For the remaining (M - j_first) side dishes, the price is P
        if j_first < M:
            sum_remaining = P * (M - j_first)
            total_price += sum_remaining
    
    return total_price

print(solve())