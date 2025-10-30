n = int(input())
s = input().strip()

if n == 0:
    print(0)
else:
    max_run = {}
    current_char = s[0]
    current_len = 1
    max_run[current_char] = 1
    
    for c in s[1:]:
        if c == current_char:
            current_len += 1
        else:
            if current_len > max_run.get(current_char, 0):
                max_run[current_char] = current_len
            current_char = c
            current_len = 1
    # Check the last run
    if current_len > max_run.get(current_char, 0):
        max_run[current_char] = current_len
    
    print(sum(max_run.values()))