# Read input values
N, M = map(int, input().split())
H = list(map(int, input().split()))

remaining = M
count = 0

for h in H:
    if remaining >= h:
        remaining -= h
        count += 1
    else:
        break

print(count)