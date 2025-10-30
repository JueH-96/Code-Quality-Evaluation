import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    color_dict = defaultdict(list)
    for _ in range(n):
        a, c = map(int, sys.stdin.readline().split())
        color_dict[c].append(a)
    
    if not color_dict:
        print(0)
        return
    
    min_list = [min(group) for group in color_dict.values()]
    print(max(min_list))

if __name__ == "__main__":
    main()