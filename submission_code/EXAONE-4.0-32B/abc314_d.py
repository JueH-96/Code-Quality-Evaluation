import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    q = int(data[2].strip())
    update_arr = [(-1, '')] * n
    last_global_time = -1
    last_global_type = None
    
    for i in range(3, 3 + q):
        parts = data[i].split()
        t = int(parts[0])
        if t == 1:
            x = int(parts[1]) - 1
            c = parts[2]
            op_index = i - 3
            update_arr[x] = (op_index, c)
        elif t == 2:
            op_index = i - 3
            last_global_time = op_index
            last_global_type = 2
        elif t == 3:
            op_index = i - 3
            last_global_time = op_index
            last_global_type = 3
            
    res = []
    base_chars = list(s)
    for i in range(n):
        op_index, c_updated = update_arr[i]
        if op_index == -1:
            ch = base_chars[i]
            if last_global_time != -1:
                if last_global_type == 2:
                    ch = ch.lower()
                else:
                    ch = ch.upper()
            res.append(ch)
        else:
            if op_index > last_global_time:
                res.append(c_updated)
            else:
                if last_global_type == 2:
                    res.append(c_updated.lower())
                else:
                    res.append(c_updated.upper())
                    
    print(''.join(res))

if __name__ == "__main__":
    main()