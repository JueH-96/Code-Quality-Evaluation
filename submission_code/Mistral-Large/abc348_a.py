import sys

input = sys.stdin.read
data = input().strip()

N = int(data)

results = []
for i in range(1, N + 1):
    if i % 3 == 0:
        results.append('x')
    else:
        results.append('o')

print(''.join(results))