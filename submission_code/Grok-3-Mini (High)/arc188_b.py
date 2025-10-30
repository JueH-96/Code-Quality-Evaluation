import math
import sys

data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1
for _ in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    index += 2
    g = math.gcd(2 * K, N)
    size = N // g
    if size % 2 == 1 and ((2 * K % N) != 0 or N == 2):
        print("Yes")
    else:
        print("No")