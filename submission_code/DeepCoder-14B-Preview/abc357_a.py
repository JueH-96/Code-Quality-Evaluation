# Read the input values
n, m = map(int, input().split())
h = list(map(int, input().split()))

current = m
count = 0

for hi in h:
    if current <= 0:
        break
    if hi <= current:
        count += 1
        current -= hi
    else:
        current = 0

print(count)