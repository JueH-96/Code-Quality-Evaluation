def can_be_expressed_as_power_of_2_and_3(N):
    # Check for factors of 2
    while N % 2 == 0:
        N //= 2

    # Check for factors of 3
    while N % 3 == 0:
        N //= 3

    # If N is reduced to 1, it means it can be expressed as 2^x * 3^y
    return N == 1

# Read input
N = int(input().strip())

# Determine if N can be expressed as 2^x * 3^y
if can_be_expressed_as_power_of_2_and_3(N):
    print("Yes")
else:
    print("No")