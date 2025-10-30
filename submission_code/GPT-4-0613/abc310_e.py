def nand(x, y):
    return 1 - x*y

N = int(input().strip())
S = list(map(int, list(input().strip())))

count = [0]*(N+1)
for i in range(N-1, -1, -1):
    if S[i] == 1:
        count[i] = count[i+1] + 1
    else:
        count[i] = count[i+1]

total = 0
for i in range(N):
    if S[i] == 0:
        total += count[i+1] + 1
    else:
        total += 2**count[i+1] - 1
print(total)