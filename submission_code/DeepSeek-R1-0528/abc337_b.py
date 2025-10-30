S = input().strip()
found_B = False
found_C = False
for char in S:
    if char == 'A':
        if found_B or found_C:
            print('No')
            break
    elif char == 'B':
        if found_C:
            print('No')
            break
        found_B = True
    else:
        found_C = True
else:
    print('Yes')