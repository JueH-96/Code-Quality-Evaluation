def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # For the sequence to be non-decreasing, we need a[0] <= a[1] <= ... <= a[n-1]
    # We can only move value from right to left
    
    # At each position i, calculate the minimum value it should have
    # based on the constraint that the sequence should be non-decreasing
    min_required = 0
    
    for i in range(n):
        min_required = max(min_required, a[i])
        
        # Calculate total available in suffix starting from i
        suffix_sum = sum(a[i:])
        # Calculate minimum required for suffix starting from i
        suffix_length = n - i
        min_required_for_suffix = min_required * suffix_length
        
        if suffix_sum < min_required_for_suffix:
            return "No"
    
    return "Yes"

t = int(input())
for _ in range(t):
    print(solve())