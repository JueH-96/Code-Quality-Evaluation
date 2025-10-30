# YOUR CODE HERE
N = int(input())

# Divide by 2 as many times as possible
while N % 2 == 0:
    N //= 2

# Divide by 3 as many times as possible
while N % 3 == 0:
    N //= 3

# If N becomes 1, then it was of the form 2^x * 3^y
if N == 1:
    print("Yes")
else:
    print("No")