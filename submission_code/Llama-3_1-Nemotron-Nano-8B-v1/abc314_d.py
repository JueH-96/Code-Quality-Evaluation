import sys

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    q = int(sys.stdin.readline())
    
    original_chars = list(s)
    mod_char = [None] * n
    mod_time = [-1] * n
    global_case = None
    global_op_count = 0
    
    for op_num in range(1, q + 1):
        parts = sys.stdin.readline().split()
        t = int(parts[0])
        x = int(parts[1])
        c = parts[2]
        
        if t == 1:
            x_idx = x - 1
            mod_char[x_idx] = c
            mod_time[x_idx] = op_num
        else:
            global_case = 'lower' if t == 2 else 'upper'
            global_op_count += 1
    
    result = []
    for i in range(n):
        if mod_time[i] == -1:
            current_char = original_chars[i]
        else:
            current_char = mod_char[i]
        
        if mod_time[i] > global_op_count:
            final_char = current_char
        else:
            if global_case is None:
                final_char = current_char
            elif global_case == 'lower':
                final_char = current_char.lower()
            else:
                final_char = current_char.upper()
        
        result.append(final_char)
    
    print(''.join(result))

if __name__ == "__main__":
    main()