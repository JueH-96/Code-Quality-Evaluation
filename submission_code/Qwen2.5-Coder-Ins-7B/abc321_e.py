import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
results = []

index = 1
for _ in range(T):
    N = int(data[index])
    X = int(data[index + 1])
    K = int(data[index + 2])
    index += 3

    if K == 0:
        results.append(1)
        continue

    count = 0
    for i in range(X, N + 1):
        if i % (2 ** K) == X % (2 ** K):
            count += 1
    results.append(count)

for result in results:
    print(result)