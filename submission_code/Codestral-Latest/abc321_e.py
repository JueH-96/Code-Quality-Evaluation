import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    X = int(data[index + 1])
    K = int(data[index + 2])
    index += 3

    if K == 0:
        results.append(1)
    else:
        left = X - (1 << (K - 1))
        right = X + (1 << (K - 1)) - 1
        if left < 1:
            left = 1
        if right > N:
            right = N
        results.append(max(0, right - left + 1))

for result in results:
    print(result)