S = input().strip()

current_section = 0  # 0: A section, 1: B section, 2: C section

for char in S:
    if current_section == 0:
        if char == 'A':
            continue
        elif char == 'B':
            current_section = 1
        elif char == 'C':
            current_section = 2
    elif current_section == 1:
        if char == 'B':
            continue
        elif char == 'C':
            current_section = 2
        elif char == 'A':
            print("No")
            exit()
    elif current_section == 2:
        if char == 'C':
            continue
        else:
            print("No")
            exit()

print("Yes")