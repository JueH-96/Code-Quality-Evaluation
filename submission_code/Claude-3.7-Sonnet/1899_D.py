from collections import defaultdict
from math import log2

def count_pairs(a):
    n = len(a)
    count = 0
    value_count = defaultdict(int)
    
    for i in range(n):
        f_val = log2(a[i]) - a[i]
        # Round to handle potential floating-point precision issues
        f_val = round(f_val, 15)
        count += value_count[f_val]
        value_count[f_val] += 1
    
    return count

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(count_pairs(a))