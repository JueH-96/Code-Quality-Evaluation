n, k = map(int, input().split())
s = input().strip()

current_char = None
current_length = 0
total = 0

for char in s:
    if char == current_char:
        current_length += 1
    else:
        if current_char == 'O':
            total += current_length // k
        current_char = char
        current_length = 1

# Add the last run
if current_char == 'O':
    total += current_length // k

print(total)