s = input().strip()
total = 0
prev = None
count = 0

for c in s:
    if c == prev:
        count += 1
    else:
        if prev is not None:
            if prev == '0':
                total += (count + 1) // 2
            else:
                total += count
        prev = c
        count = 1

# Process the last group
if prev == '0':
    total += (count + 1) // 2
else:
    total += count

print(total)