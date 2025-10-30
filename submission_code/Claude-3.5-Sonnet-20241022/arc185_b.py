def can_make_non_decreasing(N, A):
    # Convert to list to make it mutable
    A = list(A)
    
    # For each position from right to left
    for i in range(N-1, 0, -1):
        # If current element is less than previous element
        if A[i] < A[i-1]:
            # Calculate difference needed
            diff = A[i-1] - A[i]
            
            # Check if we have enough "capacity" in previous elements
            total_available = 0
            for j in range(i-1, -1, -1):
                if j > 0:
                    total_available += A[j] - A[j-1] if A[j] > A[j-1] else 0
                else:
                    total_available += A[0]
                    
            if total_available < diff:
                return False
                
            # Distribute the difference
            remaining = diff
            for j in range(i-1, -1, -1):
                if remaining == 0:
                    break
                if j > 0:
                    available = A[j] - A[j-1] if A[j] > A[j-1] else 0
                else:
                    available = A[0]
                    
                used = min(available, remaining)
                A[j] -= used
                remaining -= used
            
            A[i] += diff
            
    return True

# Read number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    # Read N and array A
    N = int(input())
    A = list(map(int, input().split()))
    
    # Print result
    print("Yes" if can_make_non_decreasing(N, A) else "No")