n = int(input())
s = input().strip()
current_chars = list(s)
modified_times = [0] * n

q = int(input())
last_case = 0  # 0: none, 2: all to lower, 3: all to upper
last_case_time = 0

for op_idx in range(1, q + 1):
    parts = input().split()
    t_i = int(parts[0])
    if t_i == 1:
        x_i = int(parts[1])
        c_i = parts[2]
        idx = x_i - 1
        current_chars[idx] = c_i
        modified_times[idx] = op_idx
    else:
        last_case = t_i
        last_case_time = op_idx

result = []
for i in range(n):
    c = current_chars[i]
    mt = modified_times[i]
    if mt > last_case_time:
        result.append(c)
    else:
        if last_case == 2:
            # Convert uppercase to lowercase
            result.append(c.lower() if c.isupper() else c)
        elif last_case == 3:
            # Convert lowercase to uppercase
            result.append(c.upper() if c.islower() else c)
        else:
            result.append(c)

print(''.join(result))