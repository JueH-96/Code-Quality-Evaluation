def is_geometric_progression(sequence):
    if len(sequence) <= 2:
        return True
    
    # Instead of comparing ratios directly, use cross multiplication
    # to avoid floating point precision issues
    # For a geometric progression: A[i+1]/A[i] = A[i+2]/A[i+1]
    # Cross multiply: A[i+1]² = A[i] * A[i+2]
    
    for i in range(len(sequence) - 2):
        if sequence[i+1] ** 2 != sequence[i] * sequence[i+2]:
            return False
    
    return True

# Read input
N = int(input())
A = list(map(int, input().split()))

# Check if A is a geometric progression
if is_geometric_progression(A):
    print("Yes")
else:
    print("No")