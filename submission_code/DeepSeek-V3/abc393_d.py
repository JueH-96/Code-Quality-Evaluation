import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    
    ones = []
    for idx, c in enumerate(S):
        if c == '1':
            ones.append(idx)
    
    m = len(ones)
    if m == 0:
        print(0)
        return
    
    median_pos = ones[m // 2]
    target_start = median_pos - m // 2
    total_swaps = 0
    for i in range(m):
        original_pos = ones[i]
        target_pos = target_start + i
        total_swaps += abs(original_pos - target_pos)
    
    print(total_swaps)

solve()