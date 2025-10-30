import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
N = int(data[index])
index += 1

bars = []
for _ in range(N):
    R = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    L = int(data[index])
    index += 1
    bars.append((R, C, L))

# Simulate the movement of bars
for _ in range(10**5):
    new_bars = []
    occupied = defaultdict(set)
    for R, C, L in bars:
        if R == H:
            new_bars.append((R, C, L))
            for j in range(L):
                occupied[R].add(C + j)
        else:
            can_move = True
            for j in range(L):
                if R + 1 in occupied and C + j in occupied[R + 1]:
                    can_move = False
                    break
            if can_move:
                new_bars.append((R + 1, C, L))
                for j in range(L):
                    occupied[R + 1].add(C + j)
            else:
                new_bars.append((R, C, L))
                for j in range(L):
                    occupied[R].add(C + j)
    bars = new_bars

# Output the final positions of the bars
for R, _, _ in bars:
    print(R)