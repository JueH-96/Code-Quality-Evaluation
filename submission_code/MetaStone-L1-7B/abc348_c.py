import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    color_dict = defaultdict(list)
    for _ in range(N):
        A, C = map(int, sys.stdin.readline().split())
        color_dict[C].append(A)
    
    max_min = -float('inf')
    for color in color_dict:
        current_min = min(color_dict[color])
        if current_min > max_min:
            max_min = current_min
    print(max_min)

if __name__ == '__main__':
    main()