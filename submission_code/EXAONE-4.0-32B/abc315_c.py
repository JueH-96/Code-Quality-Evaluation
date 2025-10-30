import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    groups = {}
    
    for i in range(1, n + 1):
        line = data[i].split()
        if not line:
            continue
        f = int(line[0])
        s = int(line[1])
        if f in groups:
            lst = groups[f]
            if s > lst[0]:
                lst[1] = lst[0]
                lst[0] = s
            elif s > lst[1]:
                lst[1] = s
        else:
            groups[f] = [s, -10**18]
    
    same_candidate = -10**18
    tops = []
    
    for f in groups:
        lst = groups[f]
        tops.append(lst[0])
        if lst[1] != -10**18:
            candidate_val = lst[0] + lst[1] // 2
            if candidate_val > same_candidate:
                same_candidate = candidate_val
                
    if len(tops) < 2:
        candidate_diff = -10**18
    else:
        tops.sort(reverse=True)
        candidate_diff = tops[0] + tops[1]
        
    ans = max(same_candidate, candidate_diff)
    print(ans)

if __name__ == "__main__":
    main()