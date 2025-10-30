n = int(input())
s = input().strip()

if n == 0:
    print(0)
    exit()

max_run = {}
current_char = s[0]
current_run_length = 1

for i in range(1, n):
    if s[i] == current_char:
        current_run_length += 1
    else:
        if current_char in max_run:
            if current_run_length > max_run[current_char]:
                max_run[current_char] = current_run_length
        else:
            max_run[current_char] = current_run_length
        current_char = s[i]
        current_run_length = 1

# Update the last run
if current_char in max_run:
    if current_run_length > max_run[current_char]:
        max_run[current_char] = current_run_length
else:
    max_run[current_char] = current_run_length

# Calculate the sum of all maximum runs
result = sum(max_run.values())

print(result)