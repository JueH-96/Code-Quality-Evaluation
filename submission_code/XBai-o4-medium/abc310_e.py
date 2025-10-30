n = int(input())
s = input().strip()

prev0 = 0
prev1 = 0
total = 0

for c in s:
    if c == '0':
        curr1 = prev0 + prev1
        curr0 = 1
    else:
        curr0 = prev1
        curr1 = prev0 + 1
    total += curr1
    prev0, prev1 = curr0, curr1

print(total)