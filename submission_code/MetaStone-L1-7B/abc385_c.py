import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    H = list(map(int, sys.stdin.readline().split()))
    groups = defaultdict(list)
    for idx in range(N):
        groups[H[idx]].append(idx + 1)  # Positions are 1-based
    
    global_max = 1
    for h in groups:
        group = groups[h]
        group_set = set(group)
        current_max = 1
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                s = group[j] - group[i]
                current = 1
                next_p = group[i] + s
                while next_p in group_set:
                    current += 1
                    next_p += s
                if current > current_max:
                    current_max = current
        if current_max > global_max:
            global_max = current_max
    print(global_max)

if __name__ == '__main__':
    main()