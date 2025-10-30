import sys
data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1
for test in range(T):
    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index+N]))
    index += N

    if N == 0:
        print(0)
        continue
    runs = 1
    for i in range(1, N):
        if A[i] != A[i-1]:
            runs += 1
    print(runs)