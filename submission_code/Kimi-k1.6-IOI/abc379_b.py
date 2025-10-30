n, k = map(int, input().split())
s = input().strip()

current_o = 0
total = 0

for c in s:
    if c == 'O':
        current_o += 1
    else:
        if current_o > 0:
            total += current_o // k
            current_o = 0

# Check remaining O's after loop
if current_o > 0:
    total += current_o // k

print(total)