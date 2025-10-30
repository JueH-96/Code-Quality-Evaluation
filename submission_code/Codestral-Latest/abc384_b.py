# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
R = int(data[1])

arcs = []
for i in range(N):
    D = int(data[2 + 2 * i])
    A = int(data[3 + 2 * i])
    arcs.append((D, A))

for D, A in arcs:
    if D == 1 and 1600 <= R <= 2799:
        R += A
    elif D == 2 and 1200 <= R <= 2399:
        R += A

print(R)