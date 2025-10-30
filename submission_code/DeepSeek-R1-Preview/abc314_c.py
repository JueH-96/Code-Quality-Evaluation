import sys
from collections import defaultdict

def main():
    n, m = map(int, sys.stdin.readline().split())
    s = list(sys.stdin.readline().strip())
    c = list(map(int, sys.stdin.readline().split()))
    
    # Preprocess the positions for each color
    positions = defaultdict(list)
    for i in range(n):
        color = c[i]
        positions[color].append(i)
    
    # Process each color from 1 to M in order
    for color in range(1, m + 1):
        pos_list = positions[color]
        k = len(pos_list)
        if k <= 1:
            continue  # No shift needed for single or zero elements
        
        # Extract the characters
        chars = [s[p] for p in pos_list]
        # Rotate right by 1
        new_chars = [chars[-1]] + chars[:-1]
        
        # Update the string list
        for p, ch in zip(pos_list, new_chars):
            s[p] = ch
    
    print(''.join(s))

if __name__ == "__main__":
    main()