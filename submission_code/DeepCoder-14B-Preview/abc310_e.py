n = int(input())
s = input().strip()

prev_0 = 0
prev_1 = 0
total = 0

for c in s:
    current = int(c)
    new_0 = 0
    new_1 = 0

    if current == 0:
        new_0 += 1
    else:
        new_1 += 1

    new_1 += prev_0  # All v=0 contribute to new_1

    if current == 0:
        new_1 += prev_1
    else:
        new_0 += prev_1

    total += new_1

    prev_0, prev_1 = new_0, new_1

print(total)