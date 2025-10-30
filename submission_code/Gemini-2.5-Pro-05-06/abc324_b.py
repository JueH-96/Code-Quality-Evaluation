# YOUR CODE HERE
N = int(input())

# Remove all factors of 2
# This loop continues as long as N is even.
# Integer division N //= 2 ensures N remains an integer.
while N % 2 == 0:
    N //= 2

# Remove all factors of 3
# This loop continues as long as N is divisible by 3.
while N % 3 == 0:
    N //= 3

# After removing all factors of 2 and 3:
# If N is 1, it means the original N was composed solely of powers of 2 and 3.
# Otherwise, N has other prime factors, or was not of the form 2^x * 3^y.
if N == 1:
    print("Yes")
else:
    print("No")