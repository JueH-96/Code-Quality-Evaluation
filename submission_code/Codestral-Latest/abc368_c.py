import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

T = 0
i = 0

while i < N:
    if H[i] <= 0:
        i += 1
        continue
    T += 1
    if T % 3 == 0:
        H[i] -= 3
    else:
        H[i] -= 1

print(T)