import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    q = int(data[2].strip())
    operations = []
    for i in range(3, 3 + q):
        parts = data[i].split()
        t_i = int(parts[0])
        if t_i == 1:
            x_i = int(parts[1])
            c_i = parts[2]
            operations.append((t_i, x_i, c_i))
        else:
            operations.append((t_i, 0, 'a'))
    
    base_char_arr = list(s)
    last_update_time = [0] * n
    
    conv_after = [None] * (q + 2)
    last_conv = None
    for i in range(q, 0, -1):
        conv_after[i] = last_conv
        t_i = operations[i-1][0]
        if last_conv is None and (t_i == 2 or t_i == 3):
            last_conv = t_i
    conv_after[0] = last_conv

    for idx in range(q):
        t_i, x_i, c_i = operations[idx]
        if t_i == 1:
            pos = x_i - 1
            if 0 <= pos < n:
                base_char_arr[pos] = c_i
                last_update_time[pos] = idx + 1
                
    res = []
    for j in range(n):
        t_val = last_update_time[j]
        base_char = base_char_arr[j]
        conv_type = conv_after[t_val]
        if conv_type is None:
            res.append(base_char)
        elif conv_type == 2:
            res.append(base_char.lower())
        else:
            res.append(base_char.upper())
            
    print(''.join(res))

if __name__ == "__main__":
    main()