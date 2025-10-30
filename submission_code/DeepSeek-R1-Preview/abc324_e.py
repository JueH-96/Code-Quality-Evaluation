import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    t = data[idx]
    idx += 1
    len_t = len(t)
    
    pre = []
    suf = []
    
    for _ in range(n):
        s = data[idx]
        idx += 1
        # Compute pre_i
        ptr = 0
        for c in s:
            if ptr < len_t and c == t[ptr]:
                ptr += 1
        pre_i = ptr
        pre.append(pre_i)
        
        # Compute suf_i
        ptr_suf = len_t - 1
        for c in reversed(s):
            if ptr_suf >= 0 and c == t[ptr_suf]:
                ptr_suf -= 1
        suf_i = len_t - ptr_suf - 1
        suf.append(suf_i)
    
    # Sort the suf array for binary search
    suf.sort()
    
    count = 0
    for p in pre:
        required = max(len_t - p, 0)
        # Find the first index where suf >= required
        idx_suf = bisect.bisect_left(suf, required)
        count += len(suf) - idx_suf
    
    print(count)

if __name__ == '__main__':
    main()