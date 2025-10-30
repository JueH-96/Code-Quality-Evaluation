import sys
data = sys.stdin.read().strip()
N = int(data)
count = 0
while N % 2 == 0:
    N = N // 2
    count += 1
print(count)