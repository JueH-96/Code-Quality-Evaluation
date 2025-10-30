n, k = map(int, input().split())
s = input().strip()

current = 0
total = 0

for c in s:
    if c == 'O':
        current += 1
    else:
        if current > 0:
            total += current // k
            current = 0
# Add the last run if it ends at the end of the string
if current > 0:
    total += current // k

print(total)