import bisect

def find_first_day():
    n, k = map(int, input().split())
    medicines = [tuple(map(int, input().split())) for _ in range(n)]
    
    # Sort the medicines based on the number of days (a_i)
    medicines.sort()
    
    a_sorted = [x[0] for x in medicines]
    b_sorted = [x[1] for x in medicines]
    
    # Compute the suffix sum of b's
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + b_sorted[i]
    
    # Determine the initial high value
    if not a_sorted:
        max_a = 0
    else:
        max_a = a_sorted[-1]
    low = 1
    high = max_a + 1  # Since X can be up to max_a + 1
    
    # Binary search to find the minimal X
    while low < high:
        mid = (low + high) // 2
        # Find the first index where a_i >= mid
        pos = bisect.bisect_left(a_sorted, mid)
        current_sum = suffix_sum[pos]
        
        if current_sum <= k:
            high = mid
        else:
            low = mid + 1
    
    print(low)

find_first_day()