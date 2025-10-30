import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    color_groups = defaultdict(list)
    for _ in range(n):
        a, c = map(int, sys.stdin.readline().split())
        color_groups[c].append(a)
    
    max_min = 0
    for beans in color_groups.values():
        current_min = min(beans)
        if current_min > max_min:
            max_min = current_min
    print(max_min)

if __name__ == "__main__":
    main()