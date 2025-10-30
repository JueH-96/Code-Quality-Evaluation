import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    color_dict = defaultdict(list)
    for _ in range(n):
        a, c = map(int, sys.stdin.readline().split())
        color_dict[c].append(a)
    
    max_min = 0
    for c in color_dict:
        current_min = min(color_dict[c])
        if current_min > max_min:
            max_min = current_min
    print(max_min)

if __name__ == "__main__":
    main()