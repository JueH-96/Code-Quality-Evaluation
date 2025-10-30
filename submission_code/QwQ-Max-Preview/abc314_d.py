n = int(input())
s = list(input())
q = int(input())

# Initialize each character's value and last_modified time
chars = [(c, 0) for c in s]
last_global = 0
global_type = 0  # 0: none, 2: type2 (lower), 3: type3 (upper)

for op_num in range(1, q + 1):
    parts = input().split()
    t = int(parts[0])
    x = int(parts[1])
    c = parts[2]
    
    if t == 1:
        idx = x - 1
        chars[idx] = (c, op_num)
    else:
        last_global = op_num
        global_type = t

result = []
for (c, lm) in chars:
    if lm >= last_global:
        result.append(c)
    else:
        if global_type == 2:
            result.append(c.lower())
        elif global_type == 3:
            result.append(c.upper())
        else:
            result.append(c)

print(''.join(result))