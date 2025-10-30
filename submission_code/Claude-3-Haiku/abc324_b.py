import math

N = int(input())

# Check if N can be expressed as 2^x * 3^y
for x in range(int(math.log2(N)) + 1):
    for y in range(int(math.log(N, 3)) + 1):
        if 2 ** x * 3 ** y == N:
            print("Yes")
            return

print("No")