def minimum_passengers(N, A):
    total_change = sum(A)
    # If total_change is negative, we need at least -total_change initial passengers
    return max(0, -total_change)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))

# Get the result and print it
result = minimum_passengers(N, A)
print(result)