def can_make_kagamimochi(a, b):
    # Returns True if mochi of size a can be placed on mochi of size b
    return a <= b//2

def max_kagamimochi(N, A):
    # Using two pointers approach
    left = 0  # pointer for smaller mochi
    right = N-1  # pointer for larger mochi
    count = 0  # count of possible kagamimochi
    
    while left < right:
        # If we can make kagamimochi with current pair
        if can_make_kagamimochi(A[left], A[right]):
            count += 1
            left += 1
            right -= 1
        else:
            # If current smallest mochi is too big for current largest mochi
            # Try with next largest mochi
            right -= 1
            
    return count

# Read input
N = int(input())
A = list(map(int, input().split()))

# Print result
print(max_kagamimochi(N, A))