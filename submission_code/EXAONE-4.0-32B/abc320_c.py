def main():
    import sys
    data = sys.stdin.read().splitlines()
    M = int(data[0].strip())
    S1 = data[1].strip()
    S2 = data[2].strip()
    S3 = data[3].strip()
    
    set1 = set(S1)
    set2 = set(S2)
    set3 = set(S3)
    common_digits = set1 & set2 & set3
    
    if not common_digits:
        print(-1)
        return
        
    pos1 = {}
    for idx, char in enumerate(S1):
        if char not in pos1:
            pos1[char] = []
        pos1[char].append(idx)
        
    pos2 = {}
    for idx, char in enumerate(S2):
        if char not in pos2:
            pos2[char] = []
        pos2[char].append(idx)
        
    pos3 = {}
    for idx, char in enumerate(S3):
        if char not in pos3:
            pos3[char] = []
        pos3[char].append(idx)
        
    best = float('inf')
    
    for d in common_digits:
        list1 = pos1[d]
        list2 = pos2[d]
        list3 = pos3[d]
        for r0 in list1:
            for r1 in list2:
                for r2 in list3:
                    arr = sorted([r0, r1, r2])
                    t0 = arr[0]
                    t1 = arr[1]
                    if t1 == t0:
                        t1 += M
                    t2 = arr[2]
                    if t2 == t0:
                        t2 += M
                    if t2 == t1:
                        t2 += M
                    total_time = max(t0, t1, t2)
                    if total_time < best:
                        best = total_time
                        
    if best == float('inf'):
        print(-1)
    else:
        print(best)

if __name__ == "__main__":
    main()