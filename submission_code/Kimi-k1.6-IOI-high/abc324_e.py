import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    T = data[1]
    m = len(T)
    strings = data[2:2+n]
    
    prefixes = []
    suffixes = []
    
    for s in strings:
        # Compute prefix
        t_ptr = 0
        for c in s:
            if t_ptr < m and c == T[t_ptr]:
                t_ptr += 1
        prefix = t_ptr
        prefixes.append(prefix)
        
        # Compute suffix
        t_ptr_suffix = m - 1
        for c in reversed(s):
            if t_ptr_suffix >= 0 and c == T[t_ptr_suffix]:
                t_ptr_suffix -= 1
        suffix = (m - 1) - t_ptr_suffix
        suffixes.append(suffix)
    
    prefixes_sorted = sorted(prefixes)
    ans = 0
    for s in suffixes:
        required = m - s
        idx = bisect.bisect_left(prefixes_sorted, required)
        ans += len(prefixes_sorted) - idx
    
    print(ans)

if __name__ == "__main__":
    main()