def find_smallest_326_like_number(N):
    # Extract digits of N
    A = N // 100
    B = (N // 10) % 10
    C = N % 10
    
    min_M = 9999  # Initialize to a large number
    
    # Iterate through possible hundreds digits starting from A
    for D in range(A, 10):
        # Iterate through possible tens digits
        for E in range(0, 10):
            F = D * E
            if F > 9:
                continue  # Ones digit must be a single digit (0-9)
            M = 100 * D + 10 * E + F
            if M >= N and M < min_M:
                min_M = M
    return min_M

# Read input
N = int(input())

# Find and print the smallest 326-like number >= N
print(find_smallest_326_like_number(N))