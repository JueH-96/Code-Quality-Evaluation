def find_closest_values(N, L, R, A):
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)
    return result

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
L = int(data[1])
R = int(data[2])
A = list(map(int, data[3:]))

result = find_closest_values(N, L, R, A)
print(" ".join(map(str, result)))