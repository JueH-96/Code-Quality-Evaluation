import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    pos = defaultdict(list)
    
    for idx, num in enumerate(a):
        pos[num].append(idx)
    
    min_len = float('inf')
    
    for key in pos:
        indices = pos[key]
        if len(indices) < 2:
            continue
        for i in range(1, len(indices)):
            current_diff = indices[i] - indices[i-1] + 1
            if current_diff < min_len:
                min_len = current_diff
    
    if min_len == float('inf'):
        print(-1)
    else:
        print(min_len)

if __name__ == "__main__":
    main()