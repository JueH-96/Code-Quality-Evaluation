n = int(input())
s = input().strip()
q = int(input())

arr = list(s)
last_modified = [0] * n
last_conv_type = 0
step_conv = -1

for step in range(1, q + 1):
    parts = input().split()
    t = int(parts[0])
    x = int(parts[1])
    c = parts[2]
    if t == 1:
        idx = x - 1
        arr[idx] = c
        last_modified[idx] = step
    else:
        last_conv_type = t
        step_conv = step

for i in range(n):
    if last_modified[i] <= step_conv:
        if last_conv_type == 2:
            arr[i] = arr[i].lower()
        elif last_conv_type == 3:
            arr[i] = arr[i].upper()

print(''.join(arr))