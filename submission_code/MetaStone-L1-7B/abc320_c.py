import sys
from collections import defaultdict

def main():
    M = int(sys.stdin.readline())
    S1 = sys.stdin.readline().strip()
    S2 = sys.stdin.readline().strip()
    S3 = sys.stdin.readline().strip()

    pos1 = defaultdict(list)
    for idx, c in enumerate(S1):
        pos1[c].append(idx)
    
    pos2 = defaultdict(list)
    for idx, c in enumerate(S2):
        pos2[c].append(idx)
    
    pos3 = defaultdict(list)
    for idx, c in enumerate(S3):
        pos3[c].append(idx)
    
    candidates = []
    
    for d in '0123456789':
        if not (pos1[d] and pos2[d] and pos3[d]):
            continue
        
        for a in pos1[d]:
            for b in pos2[d]:
                if a == b:
                    continue
                for c in pos3[d]:
                    if a == c or b == c:
                        continue
                    else:
                        current_max = max(a, b, c)
                        candidates.append(current_max)
                    else:
                        if a == b:
                            current_max = max(a + M, c)
                        elif a == c:
                            current_max = max(a + M, b)
                        elif b == c:
                            current_max = max(b + M, a)
                        else:
                            current_max = a + 2 * M
                        candidates.append(current_max)
    
    if not candidates:
        print(-1)
    else:
        print(min(candidates))

if __name__ == "__main__":
    main()