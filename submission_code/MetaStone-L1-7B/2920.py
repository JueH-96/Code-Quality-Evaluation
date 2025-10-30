import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    nums = list(map(int, sys.stdin.readline().split()))
    
    # For each value, collect the positions where it occurs
    pos_map = defaultdict(list)
    for idx, num in enumerate(nums):
        pos_map[num].append(idx)
    
    min_steps = float('inf')
    
    # Iterate through each unique value in the array
    for x in pos_map:
        positions = pos_map[x]
        m = len(positions)
        if m == 0:
            continue
        max_gap = 0
        # Compute gaps between consecutive positions, considering circular
        for i in range(m):
            prev = positions[i]
            curr = positions[(i + 1) % m]
            if i == m - 1:
                prev_next = positions[0]
            else:
                prev_next = positions[i + 1]
            # Compute the circular distance
            gap = (prev_next - prev) % n
            if gap > max_gap:
                max_gap = gap
        # Compute steps as ceil(max_gap / 2)
        t = (max_gap + 1) // 2
        if t < min_steps:
            min_steps = t
    print(min_steps)

if __name__ == "__main__":
    main()