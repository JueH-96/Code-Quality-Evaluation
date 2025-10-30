s = input().strip()

if not s:
    print(0)
else:
    runs = []
    current_char = s[0]
    current_length = 1
    for c in s[1:]:
        if c == current_char:
            current_length += 1
        else:
            runs.append((current_char, current_length))
            current_char = c
            current_length = 1
    runs.append((current_char, current_length))
    
    saved = 0
    for char, length in runs:
        if char == '0':
            saved += length // 2
    
    print(len(s) - saved)