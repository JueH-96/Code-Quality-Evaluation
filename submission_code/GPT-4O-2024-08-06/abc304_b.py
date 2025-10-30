# YOUR CODE HERE
def approximate_number(N):
    if N < 10**3:
        return N
    elif N < 10**4:
        return (N // 10) * 10
    elif N < 10**5:
        return (N // 100) * 100
    elif N < 10**6:
        return (N // 1000) * 1000
    elif N < 10**7:
        return (N // 10000) * 10000
    elif N < 10**8:
        return (N // 100000) * 100000
    elif N < 10**9:
        return (N // 1000000) * 1000000

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Print the result
print(approximate_number(N))