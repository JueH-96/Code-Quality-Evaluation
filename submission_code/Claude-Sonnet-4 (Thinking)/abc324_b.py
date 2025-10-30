N = int(input())

# Keep dividing by 2
while N % 2 == 0:
    N //= 2

# Keep dividing by 3
while N % 3 == 0:
    N //= 3

# If N is 1, then it can be expressed as 2^x * 3^y
if N == 1:
    print("Yes")
else:
    print("No")