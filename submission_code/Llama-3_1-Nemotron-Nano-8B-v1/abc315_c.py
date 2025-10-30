import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    groups = defaultdict(list)
    for _ in range(n):
        f, s = map(int, sys.stdin.readline().split())
        groups[f].append(s)
    
    same_max = 0
    for s_list in groups.values():
        s_list.sort(reverse=True)
        if len(s_list) >= 2:
            current = s_list[0] + s_list[1] // 2
            if current > same_max:
                same_max = current
    
    max_list = []
    for s_list in groups.values():
        if s_list:
            max_list.append(s_list[0])
    max_list.sort(reverse=True)
    cross_max = 0
    if len(max_list) >= 2:
        cross_max = max_list[0] + max_list[1]
    
    print(max(same_max, cross_max))

if __name__ == "__main__":
    main()