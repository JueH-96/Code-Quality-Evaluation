def solve():
    # Read input
    N, S = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Double the array to handle wrapping around
    A = A + A
    
    # Use sliding window technique
    left = 0
    curr_sum = 0
    
    # For each possible starting position
    for right in range(len(A)):
        # Add current element to sum
        curr_sum += A[right]
        
        # Shrink window from left if sum becomes too large
        while curr_sum > S and left <= right:
            curr_sum -= A[left]
            left += 1
            
        # Check if we found the target sum
        if curr_sum == S:
            print("Yes")
            return
            
        # If window size becomes >= N, remove leftmost element
        if right - left + 1 >= N:
            curr_sum -= A[left]
            left += 1
    
    # If we get here, no solution was found
    print("No")

# Run the solver
solve()