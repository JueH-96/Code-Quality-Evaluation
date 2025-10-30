import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    # Compute prefix sums
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = P[i-1] + A[i-1]
    
    mod_P = [p % M for p in P]
    P_total_mod = mod_P[-1]
    
    # Build the indices dictionary
    indices = defaultdict(list)
    for idx, val in enumerate(mod_P):
        indices[val].append(idx)
    
    total = 0
    for s in range(1, N + 1):
        i = s - 1
        target1 = mod_P[i]
        lst = indices.get(target1, [])
        # Calculate count1: j >= i+1 and j <= N-1
        left = bisect.bisect_left(lst, i + 1)
        right = bisect.bisect_right(lst, N - 1)
        count1 = right - left
        
        # Calculate target2
        target2 = (mod_P[i] - P_total_mod) % M
        lst2 = indices.get(target2, [])
        max_j = i - 1
        # Calculate count2: j <= max_j
        count2 = bisect.bisect_right(lst2, max_j)
        
        total += count1 + count2
    
    print(total)

if __name__ == '__main__':
    main()