def max_alternating_subarray_sum(n, arr):
    # We will use two variables to track the maximum sum of subarrays ending with an odd or even element
    max_odd = float('-inf')
    max_even = float('-inf')
    
    for value in arr:
        if value % 2 == 0:
            # Current value is even
            new_max_even = max(max_even + value, value)  # Extend even subarray or start new with current even
            new_max_odd = max(max_odd, max_odd + value)  # Extend odd subarray with even value
        else:
            # Current value is odd
            new_max_odd = max(max_odd + value, value)  # Extend odd subarray or start new with current odd
            new_max_even = max(max_even, max_even + value)  # Extend even subarray with odd value
        
        # Update the max values
        max_odd = new_max_odd
        max_even = new_max_even
    
    # The result for this test case is the maximum of the two
    return max(max_odd, max_even)

import sys
input = sys.stdin.read
data = input().split()

index = 0
t = int(data[index])
index += 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    arr = list(map(int, data[index:index+n]))
    index += n
    results.append(max_alternating_subarray_sum(n, arr))

for result in results:
    print(result)