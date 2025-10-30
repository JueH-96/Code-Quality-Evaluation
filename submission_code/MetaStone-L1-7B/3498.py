import sys
from collections import defaultdict

def main():
    n, k = map(int, sys.stdin.readline().split())
    nums = list(map(int, sys.stdin.readline().split()))
    pairs = []
    for i in range(n // 2):
        a = nums[i]
        b = nums[n - 1 - i]
        d = abs(a - b)
        m = max(k - b, a)
        pairs.append((d, m))
    
    max_m = max(m for d, m in pairs) if pairs else 0
    max_possible_X = max(d for d, m in pairs) if pairs else 0
    X_range = max(max_possible_X, max_m) + 1
    
    # Compute sum_ge: sum of m >= X
    sum_ge = [0] * (X_range + 2)
    for d, m in pairs:
        sum_ge[m] += 1
    for X in range(X_range - 1, -1, -1):
        sum_ge[X] += sum_ge[X + 1]
    
    # Compute cnt_eq: count of d == X
    cnt_eq = defaultdict(int)
    for d, m in pairs:
        cnt_eq[d] += 1
    
    min_changes = float('inf')
    for X in range(0, X_range + 1):
        # count_1: sum_ge[X] - cnt_eq[X]
        # count_2: sum_ge[X+1]
        count_1 = sum_ge[X] - cnt_eq.get(X, 0)
        count_2 = sum_ge[X + 1]
        total = count_1 + 2 * count_2
        if total < min_changes:
            min_changes = total
    
    print(min_changes)

if __name__ == "__main__":
    main()