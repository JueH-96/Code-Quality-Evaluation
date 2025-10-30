import sys

N = int(input())

# Count the number of consecutive zeros at the end of the binary representation of N
ctz = 0
while N % 2 == 0:
    ctz += 1
    N //= 2

print(ctz)