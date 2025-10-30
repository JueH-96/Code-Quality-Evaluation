import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    x = int(data[1])
    arr = list(map(int, data[2:2 + n - 1]))
    
    total_known = sum(arr)
    min_known = min(arr)
    max_known = max(arr)
    
    candidates = []
    
    if total_known - max_known >= x:
        candidates.append(0)
        
    if total_known - min_known >= x:
        candidates.append(max_known)
        
    if min_known < max_known:
        base = total_known - min_known - max_known
        s_required = x - base
        s_candidate = max(min_known + 1, s_required)
        if s_candidate < max_known:
            candidates.append(s_candidate)
            
    if candidates:
        print(min(candidates))
    else:
        print(-1)

if __name__ == "__main__":
    main()