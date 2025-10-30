def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = int(input[ptr])
    ptr += 1
    B = int(input[ptr])
    ptr += 1
    W = A + B
    D = list(map(int, input[ptr:ptr+N]))
    
    intervals = []
    for d in D:
        s = (A - d) % W
        e = (s + B - 1) % W
        if s <= e:
            intervals.append((s, e))
        else:
            intervals.append((s, W - 1))
            intervals.append((0, e))
    
    # Sort intervals by their start
    intervals.sort()
    
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last_s, last_e = merged[-1]
            curr_s, curr_e = interval
            if curr_s <= last_e + 1:
                # Merge the intervals
                new_s = last_s
                new_e = max(last_e, curr_e)
                merged[-1] = (new_s, new_e)
            else:
                merged.append((curr_s, curr_e))
    
    total = 0
    for s, e in merged:
        total += e - s + 1
    
    if total == W:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()