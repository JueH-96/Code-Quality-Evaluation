S = input().strip()

total_presses = 0
zero_count = 0

for char in S:
    if char == '0':
        zero_count += 1
    else:
        if zero_count > 0:
            total_presses += (zero_count + 1) // 2
            zero_count = 0
        total_presses += 1  # For the non-zero digit

# Handle trailing zeros
if zero_count > 0:
    total_presses += (zero_count + 1) // 2

print(total_presses)