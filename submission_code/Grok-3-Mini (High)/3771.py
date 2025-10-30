class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        
        n = len(s)
        first_occ = [-1] * 26
        last_occ = [-1] * 26
        
        # Compute first and last occurrence for each character
        for i in range(n):
            idx = ord(s[i]) - ord('a')
            if first_occ[idx] == -1:
                first_occ[idx] = i
            last_occ[idx] = i  # Update last occurrence each time
        
        # Collect all possible L and R candidates
        L_candidates = [first_occ[idx] for idx in range(26) if first_occ[idx] != -1]
        R_candidates = [last_occ[idx] for idx in range(26) if last_occ[idx] != -1]
        
        special_intervals = set()  # Store unique (start, end) tuples
        
        # Check all pairs of L_val and R_val
        for L_val in L_candidates:
            for R_val in R_candidates:
                if L_val <= R_val:
                    # Find the set of characters that appear in [L_val, R_val]
                    char_set = set()
                    for i in range(L_val, R_val + 1):
                        idx_char = ord(s[i]) - ord('a')
                        char_set.add(idx_char)
                    
                    # Check if all characters in the set have their first and last occurrence within [L_val, R_val]
                    all_good = True
                    for char_idx in char_set:
                        if first_occ[char_idx] < L_val or last_occ[char_idx] > R_val:
                            all_good = False
                            break
                    
                    # If all good and not the whole string, add to special intervals
                    if all_good and not (L_val == 0 and R_val == n - 1):
                        special_intervals.add((L_val, R_val))
        
        # Convert set to list for sorting
        intervals_list = list(special_intervals)
        
        # Sort intervals by end time
        intervals_list.sort(key=lambda x: x[1])
        
        # Greedy algorithm to find the maximum number of disjoint intervals
        count = 0
        last_end = -1
        for start, end in intervals_list:
            if start >= last_end:
                count += 1
                last_end = end
        
        # Return true if the maximum number is at least k
        return count >= k