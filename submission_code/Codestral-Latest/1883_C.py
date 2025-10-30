import sys
input = sys.stdin.read
data = input().split()

index = 0
t = int(data[index])
index += 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    index += 2
    a = list(map(int, data[index:index + n]))
    index += n

    if k == 2:
        count = sum(1 for x in a if x % 2 == 0)
        results.append(n - count)
    elif k == 3:
        count = sum(1 for x in a if x % 3 == 0)
        results.append(n - count)
    elif k == 4:
        count = sum(1 for x in a if x % 4 == 0)
        results.append(n - count)
    elif k == 5:
        count = sum(1 for x in a if x % 5 == 0)
        results.append(n - count)

for result in results:
    print(result)