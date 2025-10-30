from collections import defaultdict
import sys

input = sys.stdin.readline

def main():
    N = int(input())
    AC = [list(map(int, input().split())) for _ in range(N)]
    
    color_to_deliciousness = defaultdict(list)
    for a, c in AC:
        color_to_deliciousness[c].append(a)
    
    max_min_deliciousness = 0
    for c, a_list in color_to_deliciousness.items():
        if len(a_list) > 1:
            a_list.sort()
            max_min_deliciousness = max(max_min_deliciousness, a_list[-2])
        else:
            max_min_deliciousness = max(max_min_deliciousness, a_list[0])
    
    print(max_min_deliciousness)

if __name__ == "__main__":
    main()