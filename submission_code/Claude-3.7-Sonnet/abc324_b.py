# Read input
N = int(input())

# Remove all factors of 2
while N % 2 == 0:
    N //= 2

# Remove all factors of 3
while N % 3 == 0:
    N //= 3

# Check if N is now 1
if N == 1:
    print("Yes")
else:
    print("No")