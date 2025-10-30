import sys

def main():
    import heapq
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    
    pull = []
    regular = []
    opener = []
    
    for _ in range(N):
        t = int(data[idx])
        x = int(data[idx+1])
        idx +=2
        if t == 0:
            pull.append(x)
        elif t ==1:
            regular.append(x)
        else:
            opener.append(x)
    
    # Sort all in descending order
    pull.sort(reverse=True)
    regular.sort(reverse=True)
    opener.sort(reverse=True)
    
    # Compute prefix sums
    pull_prefix = [0]
    for x in pull:
        pull_prefix.append(pull_prefix[-1] + x)
    
    reg_prefix = [0]
    for x in regular:
        reg_prefix.append(reg_prefix[-1] + x)
    
    opener_prefix = [0]
    for x in opener:
        opener_prefix.append(opener_prefix[-1] + x)
    
    len_pull = len(pull_prefix) -1
    len_reg = len(reg_prefix) -1
    len_opener = len(opener_prefix) -1
    
    max_total = 0
    
    # Try all possible K in 0 ... min(M, len opener)
    max_k = min(M, len_opener)
    
    for K in range(0, max_k +1):
        T = M - K
        if T <0:
            continue
        # opener capacity
        S = opener_prefix[K]
        # best r is up to min(S, T, len_reg)
        r_max = min(S, T, len_reg)
        p_available = T
        p_max = min(T, len_pull)
        
        # Find best r in [0, r_max], p = T -r, p <= p_max and p >=0
        best_sum = 0
        
        # Try several points to find best_sum
        # Try r in several key positions
        candidates = []
        candidates.append(0)
        candidates.append(r_max)
        candidates.append(min(r_max, T))
        candidates.append(min(r_max, T//2))
        candidates.append(min(r_max, T -0))
        
        # Also try around the possible optimal merge of T elements
        r_merge = min(T, len_reg)
        p_merge = T - r_merge
        if p_merge >=0 and p_merge <= len_pull:
            candidates.append(r_merge)
        
        for r in candidates:
            p = T - r
            if p <0 or r <0:
                continue
            if r > len_reg or p > len_pull:
                continue
            current = reg_prefix[r] + pull_prefix[p]
            if current > best_sum:
                best_sum = current
        
        # Also try binary search area
        # Try r between max(0, r_max - 100) to r_max
        for dr in range(-100, 1):
            r = r_max + dr
            if r <0 or r > r_max:
                continue
            p = T - r
            if p <0 or p > len_pull:
                continue
            current = reg_prefix[r] + pull_prefix[p]
            if current > best_sum:
                best_sum = current
        
        # Also try p_merge
        for p in range(max(0, T - r_max -1), min(len_pull, T) +1):
            r = T - p
            if r <0 or r > r_max or r > len_reg:
                continue
            current = reg_prefix[r] + pull_prefix[p]
            if current > best_sum:
                best_sum = current
        
        max_total = max(max_total, best_sum)
    
    print(max_total)

if __name__ == "__main__":
    main()