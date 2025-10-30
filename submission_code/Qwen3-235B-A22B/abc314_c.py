import sys
from collections import defaultdict

def main():
    n, m = map(int, sys.stdin.readline().split())
    s = list(sys.stdin.readline().strip())
    c = list(map(int, sys.stdin.readline().split()))
    
    color_dict = defaultdict(list)
    for idx in range(n):
        color = c[idx]
        color_dict[color].append(idx)
    
    for color in range(1, m+1):
        positions = color_dict[color]
        # Since each color has at least one position, no need to check for empty
        chars = [s[p] for p in positions]
        rotated = [chars[-1]] + chars[:-1]
        for i in range(len(positions)):
            s[positions[i]] = rotated[i]
    
    print(''.join(s))

if __name__ == "__main__":
    main()