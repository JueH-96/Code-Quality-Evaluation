# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total = 0
    
    for L in range(n):
        # Use a set to track distinct values in current subarray
        distinct = set()
        
        for R in range(L, n):
            # Add current element to the set
            distinct.add(a[R])
            
            # Calculate f(L, R)
            # Sort the distinct values
            sorted_values = sorted(distinct)
            
            # Count the number of contiguous ranges
            if len(sorted_values) == 0:
                operations = 0
            else:
                operations = 1
                for i in range(1, len(sorted_values)):
                    # If there's a gap between consecutive values
                    if sorted_values[i] - sorted_values[i-1] > 1:
                        operations += 1
            
            total += operations
    
    print(total)

solve()