import sys
from collections import defaultdict

def main():
    S = sys.stdin.readline().strip()
    positions = defaultdict(list)
    for idx, c in enumerate(S):
        positions[c].append(idx + 1)  # Use 1-based indexing
    ans = 0
    for c in positions:
        lst = positions[c]
        m = len(lst)
        if m < 2:
            continue
        sum_formula = 0
        for k in range(1, m + 1):
            coeff = 2 * k - 1 - m
            sum_formula += lst[k-1] * coeff
        count_pairs = m * (m - 1) // 2
        ans += (sum_formula - count_pairs)
    print(ans)

if __name__ == "__main__":
    main()