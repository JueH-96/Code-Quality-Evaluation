import bisect

# Read input
n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the lists for binary search
A.sort()
B.sort()

low = 1
high = 10**18

# Binary search to find the minimal X
while low < high:
    mid = (low + high) // 2
    # Calculate the number of sellers who can sell at mid or lower
    count_sellers = bisect.bisect_right(A, mid)
    # Calculate the number of buyers who can buy at mid or higher
    count_buyers = len(B) - bisect.bisect_left(B, mid)
    
    if count_sellers >= count_buyers:
        high = mid
    else:
        low = mid + 1

print(low)