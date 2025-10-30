# Read the integer N from standard input.
# The constraints guarantee a valid positive integer is provided.
N = int(input())

# A positive integer N is of the form 2^x * 3^y if and only if its
# only prime factors are 2 and 3. We can test this by repeatedly
# dividing out factors of 2 and 3. If the result is 1, the
# condition holds.

# Remove all factors of 2 from N.
while N % 2 == 0:
    N //= 2

# Remove all factors of 3 from N.
while N % 3 == 0:
    N //= 3

# If N has been reduced to 1, it was composed solely of factors of 2 and 3.
if N == 1:
    print("Yes")
else:
    # Otherwise, N has other prime factors remaining.
    print("No")