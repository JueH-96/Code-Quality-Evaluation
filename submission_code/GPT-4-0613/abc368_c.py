import sys
from math import ceil

N = int(sys.stdin.readline())
H = list(map(int, sys.stdin.readline().split()))

T = 0
for h in H:
    T += 2 * (h // 3) + ceil(h % 3 / 2)
print(T)