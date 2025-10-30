import sys
from collections import defaultdict

def main():
    s = sys.stdin.readline().strip()
    positions = defaultdict(list)
    for idx, c in enumerate(s, 1):  # 1-based index
        positions[c].append(idx)
    
    total = 0
    for c in positions:
        lst = positions[c]
        m = len(lst)
        if m < 2:
            continue
        # Compute prefix sums
        prefix = [0] * (m + 1)
        for i in range(1, m + 1):
            prefix[i] = prefix[i-1] + lst[i-1]
        sum_contrib = 0
        for j in range(1, m):
            current = lst[j] * j - prefix[j]
            sum_contrib += current
        pairs = m * (m - 1) // 2
        total += (sum_contrib - pairs)
    print(total)

if __name__ == "__main__":
    main()