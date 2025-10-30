import sys
import bisect

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    
    first_line = data[0].split()
    n = int(first_line[0])
    T_str = first_line[1].strip()
    
    strings = []
    for i in range(1, n + 1):
        strings.append(data[i].strip())
    
    L_list = []
    H_list = []
    len_T = len(T_str)
    
    for s in strings:
        ptr = 0
        for char in s:
            if ptr < len_T and char == T_str[ptr]:
                ptr += 1
        L_list.append(ptr)
        
        t_ptr = len_T - 1
        for i in range(len(s) - 1, -1, -1):
            if t_ptr >= 0 and s[i] == T_str[t_ptr]:
                t_ptr -= 1
        H_list.append(t_ptr + 1)
    
    L_list.sort()
    total_pairs = 0
    for h_val in H_list:
        idx = bisect.bisect_left(L_list, h_val)
        total_pairs += len(L_list) - idx
        
    print(total_pairs)

if __name__ == "__main__":
    main()