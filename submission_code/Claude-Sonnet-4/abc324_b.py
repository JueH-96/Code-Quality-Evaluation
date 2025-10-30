# YOUR CODE HERE
N = int(input())

# Remove all factors of 2
while N % 2 == 0:
    N //= 2

# Remove all factors of 3
while N % 3 == 0:
    N //= 3

# If N becomes 1, then it was composed only of powers of 2 and 3
if N == 1:
    print("Yes")
else:
    print("No")