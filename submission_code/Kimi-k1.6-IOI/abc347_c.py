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
        s = (1 - d + A) % W
        e = (-d) % W
        if s <= e:
            intervals.append((s, e))
        else:
            intervals.append((s, W-1))
            intervals.append((0, e))
    
    # Sort intervals by start
    intervals.sort()
    
    merged = []
    for s, e in intervals:
        if not merged:
            merged.append((s, e))
        else:
            last_s, last_e = merged[-1]
            if s <= last_e + 1:
                # Merge
                new_s = last_s
                new_e = max(last_e, e)
                merged[-1] = (new_s, new_e)
            else:
                merged.append((s, e))
    
    total = 0
    for s, e in merged:
        total += e - s + 1
    
    if total < W:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()