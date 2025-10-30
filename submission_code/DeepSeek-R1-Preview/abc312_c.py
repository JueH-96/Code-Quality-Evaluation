import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

low = 0
high = 10**18  # A sufficiently large number to cover possible X values

while low < high:
    mid = (low + high) // 2
    # Calculate the number of sellers (A_i <= mid)
    count_sellers = bisect.bisect_right(A, mid)
    # Calculate the number of buyers (B_i >= mid)
    count_buyers = len(B) - bisect.bisect_left(B, mid)
    
    # Check if the condition is met
    if count_sellers >= count_buyers:
        high = mid
    else:
        low = mid + 1

print(low)