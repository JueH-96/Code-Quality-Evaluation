import sys

def cumulative_up_to(D):
    if D == 1:
        return 10
    elif D % 2 == 0:
        M = D // 2 - 1
        S_D = (20 * (10 ** M) - 11) // 9
        return 10 + 9 * S_D
    else:  # D odd, >=3
        M = (D - 1) // 2
        S_D = (110 * (10 ** (M - 1)) - 11) // 9
        return 10 + 9 * S_D

def find_digit(N):
    left = 1
    right = 100  # Sufficiently large upper bound
    while left < right:
        mid = (left + right) // 2
        if cumulative_up_to(mid) >= N:
            right = mid
        else:
            left = mid + 1
    return left  # Smallest D with cumulative_up_to(D) >= N

# Read input from stdin
data = sys.stdin.read().strip()
N = int(data)

# Find the number of digits D for the N-th palindrome
D = find_digit(N)

if D == 1:
    # For 1-digit palindromes, directly output N-1
    print(N - 1)
else:
    # Compute the cumulative up to D-1 digits
    C_prev = cumulative_up_to(D - 1)
    # Find the position K within D-digit palindromes
    K = N - C_prev
    # Compute L = ceil(D/2)
    L = (D + 1) // 2
    # Compute floor(D/2)
    floor_D_2 = D // 2
    # Compute M, the K-th number with L digits starting from 10^(L-1)
    M = (10 ** (L - 1)) + (K - 1)
    # Convert M to string
    S_str = str(M)
    # The part to reverse is the first floor(D/2) digits
    part_to_reverse = S_str[:floor_D_2]
    # Reverse that part
    reversed_part = part_to_reverse[::-1]
    # Construct the palindrome string
    pal_str = S_str + reversed_part
    # Output the palindrome as a string
    print(pal_str)