import sys
from collections import defaultdict

def decompose(s):
    y = 0
    while s % 2 == 0:
        s //= 2
        y += 1
    return (s, y)

def main():
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    
    groups = defaultdict(lambda: defaultdict(int))  # groups[x][y] += count
    
    for _ in range(n):
        s = int(input[idx])
        c = int(input[idx + 1])
        idx += 2
        x, y = decompose(s)
        groups[x][y] += c
    
    ans = 0
    
    for x in groups:
        y_counts = defaultdict(int)
        # Populate y_counts with current group's data
        for y_in_group, cnt in groups[x].items():
            y_counts[y_in_group] += cnt
        
        # Process each y from 0 to 60 inclusive
        for current_y in range(61):  # 0 to 60
            current_count = y_counts[current_y]
            pairs = current_count // 2
            remainder = current_count % 2
            y_counts[current_y] = remainder
            y_counts[current_y + 1] += pairs
        
        # Sum all remaining counts in this group
        ans += sum(y_counts.values())
    
    print(ans)

if __name__ == "__main__":
    main()