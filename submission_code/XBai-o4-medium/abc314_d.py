import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    q = int(sys.stdin.readline())
    
    s_list = list(s)
    last_modify_time = [0] * n
    last_global_time = -1
    last_global_type = None
    
    for step in range(q):
        current_time = step + 1
        parts = sys.stdin.readline().split()
        t = int(parts[0])
        if t == 1:
            x = int(parts[1])
            c = parts[2]
            idx = x - 1
            s_list[idx] = c
            last_modify_time[idx] = current_time
        else:
            last_global_type = t
            last_global_time = current_time
    
    if last_global_time == -1:
        print(''.join(s_list))
    else:
        res = []
        for i in range(n):
            if last_modify_time[i] > last_global_time:
                res.append(s_list[i])
            else:
                if last_global_type == 2:
                    res.append(s_list[i].lower())
                else:
                    res.append(s_list[i].upper())
        print(''.join(res))
        
if __name__ == "__main__":
    main()