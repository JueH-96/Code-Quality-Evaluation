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
    for i in range(1, 1 + n):
        strings.append(data[i].strip())
    
    len_T = len(T_str)
    
    P = []
    for s in strings:
        pos = 0
        for c in s:
            if pos < len_T and c == T_str[pos]:
                pos += 1
        P.append(pos)
        
    k_min_list = []
    for s in strings:
        k = len_T - 1
        for c in reversed(s):
            if k < 0:
                break
            if c == T_str[k]:
                k -= 1
        k_min_list.append(k + 1)
        
    total = 0
    count_full = 0
    for p in P:
        if p >= len_T:
            count_full += 1
    total += count_full * n
    
    k_min_list.sort()
    
    for p_val in P:
        if p_val < len_T:
            idx = bisect.bisect_right(k_min_list, p_val)
            total += idx
            
    print(total)

if __name__ == "__main__":
    main()