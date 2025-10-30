n = int(input())
s = input().strip()

if n == 0:
    print(0)
    exit()

max_run = {}
current_char = s[0]
current_length = 1

for c in s[1:]:
    if c == current_char:
        current_length += 1
    else:
        if current_char in max_run:
            prev_max = max_run[current_char]
            contribution = current_length - prev_max
            if contribution > 0:
                max_run[current_char] += contribution
        else:
            max_run[current_char] = current_length
        current_char = c
        current_length = 1

# Process the last run
if current_char in max_run:
    prev_max = max_run[current_char]
    contribution = current_length - prev_max
    if contribution > 0:
        max_run[current_char] += contribution
else:
    max_run[current_char] = current_length

print(sum(max_run.values()))