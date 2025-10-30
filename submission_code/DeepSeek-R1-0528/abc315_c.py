import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    groups = {}
    index = 1
    for i in range(n):
        f = int(data[index])
        s = int(data[index+1])
        index += 2
        if f not in groups:
            groups[f] = []
        groups[f].append(s)
    
    same_candidate = -10**18
    max1 = -10**18
    max2 = -10**18
    
    for lst in groups.values():
        t1 = -10**18
        t2 = -10**18
        for val in lst:
            if val > t1:
                t2 = t1
                t1 = val
            elif val > t2:
                t2 = val
                
        if t2 != -10**18:
            cand = t1 + t2 // 2
            if cand > same_candidate:
                same_candidate = cand
                
        if t1 > max1:
            max2 = max1
            max1 = t1
        elif t1 > max2:
            max2 = t1
            
    candidate_diff = -10**18
    if max1 != -10**18 and max2 != -10**18:
        candidate_diff = max1 + max2
        
    ans = max(same_candidate, candidate_diff)
    print(ans)

if __name__ == "__main__":
    main()