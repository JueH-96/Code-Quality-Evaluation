def minimum_tshirts_needed(N, M, S):
    logo_needed = 0
    plain_needed = 0
    worn_plain = 0
    worn_logo = 0

    for day in S:
        if day == '1':  # Going out for a meal
            if worn_plain > 0:
                worn_plain -= 1  # Use a plain T-shirt
            else:
                plain_needed += 1  # Need a new plain T-shirt
        elif day == '2':  # Attending a competitive programming event
            if worn_logo > 0:
                worn_logo -= 1  # Use a logo T-shirt
            else:
                logo_needed += 1  # Need a new logo T-shirt
        elif day == '0':  # No plans
            # All worn T-shirts can be washed
            worn_plain = M  # All plain T-shirts are available again
            worn_logo = logo_needed  # All logo T-shirts needed are available again

    # Calculate total logo T-shirts needed
    total_logo_needed = max(0, logo_needed - M)
    return total_logo_needed

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
S = data[2]

# Get the result and print it
result = minimum_tshirts_needed(N, M, S)
print(result)