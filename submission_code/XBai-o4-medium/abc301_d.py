s = input().strip()
n = int(input())

length = len(s)

# Precompute max_possible and min_possible for each position
max_possible = [0] * length
min_possible = [0] * length

for i in range(length):
    max_str = ''
    min_str = ''
    for j in range(i, length):
        if s[j] == '?':
            max_str += '1'
            min_str += '0'
        else:
            max_str += s[j]
            min_str += s[j]
    max_possible[i] = int(max_str, 2)
    min_possible[i] = int(min_str, 2)

current_val = 0
for i in range(length):
    remaining_bits = length - i - 1
    current_char = s[i]
    possible_choices = []
    if current_char == '?':
        possible_choices = [1, 0]
    else:
        possible_choices = [int(current_char)]
    found = False
    for choice in possible_choices:
        bit_value = 1 << remaining_bits
        new_val = current_val + (choice * bit_value)
        if i == length - 1:
            possible_min = new_val
            possible_max = new_val
        else:
            min_remaining = min_possible[i+1]
            max_remaining = max_possible[i+1]
            possible_min = new_val + min_remaining
            possible_max = new_val + max_remaining
        if possible_max < n:
            current_val = new_val
            found = True
            break
        elif possible_min > n:
            continue
        else:
            current_val = new_val
            found = True
            break
    if not found:
        print(-1)
        exit()

print(current_val)