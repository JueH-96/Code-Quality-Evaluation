from collections import defaultdict

def maximumLength(nums: list[int], k: int) -> int:
    dp = defaultdict(int)
    dp[(None, None)] = 0  # Initial state: no elements, length 0
    
    for x in nums:
        tmp = defaultdict(int)
        for (v, c) in dp:
            current_len = dp[(v, c)]
            
            # Option 1: exclude the current element x
            key = (v, c)
            if current_len > tmp[key]:
                tmp[key] = current_len
            
            # Option 2: include the current element x
            if x == v:
                new_c = c
                new_len = current_len + 1
                key = (v, new_c)
                if new_len > tmp[key]:
                    tmp[key] = new_len
            else:
                if c + 1 <= k:
                    new_c = c + 1
                    new_len = current_len + 1
                    key = (x, new_c)
                    if new_len > tmp[key]:
                        tmp[key] = new_len
        
        # Merge tmp into dp, taking the maximum length for each state
        for (state, length) in tmp.items():
            if state in dp:
                if length > dp[state]:
                    dp[state] = length
            else:
                dp[state] = length
    
    # Find the maximum length across all possible states
    max_len = 0
    for (v, c), length in dp.items():
        if length > max_len:
            max_len = length
    return max_len