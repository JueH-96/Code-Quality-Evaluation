def count_full_moon_days(N, M, P):
    if M > N:
        return 0
    
    # Calculate the first full moon day that is within the range
    first_full_moon_day = M
    # Calculate the last full moon day that is within the range
    last_full_moon_day = N
    
    # Count the number of full moon days
    count = 0
    for day in range(first_full_moon_day, last_full_moon_day + 1, P):
        count += 1
    
    return count

# Read input
import sys
input = sys.stdin.read
N, M, P = map(int, input().strip().split())

# Get the result and print it
result = count_full_moon_days(N, M, P)
print(result)