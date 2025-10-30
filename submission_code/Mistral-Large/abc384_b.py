import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
R = int(data[1])

for i in range(N):
    D_i = int(data[2 + 2 * i])
    A_i = int(data[3 + 2 * i])

    if (D_i == 1 and 1600 <= R <= 2799) or (D_i == 2 and 1200 <= R <= 2399):
        R += A_i

print(R)