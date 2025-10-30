def can_make_all_ones(N, A):
    # If all elements are already 1, any string is good
    if all(x == 1 for x in A):
        return True
    
    # Try to construct a string that works
    # We'll use pattern "ARC" repeated N//3 times and handle remainder
    base = "ARC"
    if N % 3 == 0:
        # If N is divisible by 3, we can use pattern "ARC" repeated N//3 times
        # This allows us to convert any two consecutive 0s to 1s
        # as long as they're not at distance 3 from each other
        
        # Check if there are any 0s at distance 3 from each other
        for i in range(N):
            if A[i] == 0:
                # Check if there's another 0 at distance 3
                if A[(i + 3) % N] == 0:
                    return False
        return True
    
    # If N is not divisible by 3, we can always construct a string
    # that allows us to convert any two consecutive 0s to 1s
    # by using pattern "ARC" and adding extra characters
    
    # First, check if there are any consecutive 0s that can't be handled
    for i in range(N):
        if A[i] == 0 and A[(i + 1) % N] == 0 and A[(i + 2) % N] == 0:
            return False
    
    return True

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and output result
result = can_make_all_ones(N, A)
print("Yes" if result else "No")