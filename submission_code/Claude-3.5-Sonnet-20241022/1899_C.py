def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # For each position, find max sum ending at that position with even/odd last number
    max_even = [-float('inf')] * n  # max sum ending with even number
    max_odd = [-float('inf')] * n   # max sum ending with odd number
    
    # Initialize first position
    if a[0] % 2 == 0:
        max_even[0] = a[0]
    else:
        max_odd[0] = a[0]
        
    # For each position, extend previous solutions or start new subarray
    for i in range(1, n):
        if a[i] % 2 == 0:  # Current number is even
            max_even[i] = a[i]  # Start new subarray
            if max_odd[i-1] != -float('inf'):
                max_even[i] = max(max_even[i], max_odd[i-1] + a[i])  # Extend odd ending subarray
        else:  # Current number is odd
            max_odd[i] = a[i]  # Start new subarray
            if max_even[i-1] != -float('inf'):
                max_odd[i] = max(max_odd[i], max_even[i-1] + a[i])  # Extend even ending subarray
                
    # Find maximum among all valid subarrays
    result = max(max(max_even), max(max_odd))
    return result

t = int(input())
for _ in range(t):
    print(solve())