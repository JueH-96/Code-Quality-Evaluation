# Read input
N = int(input())
A = list(map(int, input().split()))

# For a sequence to be geometric progression, the ratio between
# consecutive terms should be constant
def is_geometric_progression(arr, n):
    if n < 2:
        return True
    
    # Calculate the ratio between first two numbers
    ratio = arr[1] / arr[0]
    
    # Check if the same ratio exists between all consecutive numbers
    for i in range(1, n-1):
        if arr[i+1] / arr[i] != ratio:
            return False
    
    return True

# Print result
if is_geometric_progression(A, N):
    print("Yes")
else:
    print("No")