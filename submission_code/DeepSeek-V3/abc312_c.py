def find_min_x(N, M, A, B):
    A.sort()
    B.sort()
    
    left = 1
    right = 10**18
    answer = right
    
    while left <= right:
        mid = (left + right) // 2
        # Count sellers who can sell for <= mid
        low = 0
        high = N - 1
        seller_count = 0
        while low <= high:
            mid_seller = (low + high) // 2
            if A[mid_seller] <= mid:
                seller_count = mid_seller + 1
                low = mid_seller + 1
            else:
                high = mid_seller - 1
        # Count buyers who can buy for >= mid
        low_buyer = 0
        high_buyer = M - 1
        buyer_count = 0
        while low_buyer <= high_buyer:
            mid_buyer = (low_buyer + high_buyer) // 2
            if B[mid_buyer] >= mid:
                buyer_count = M - mid_buyer
                high_buyer = mid_buyer - 1
            else:
                low_buyer = mid_buyer + 1
        if seller_count >= buyer_count:
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    return answer

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Find the minimum X
result = find_min_x(N, M, A, B)

# Print the result
print(result)