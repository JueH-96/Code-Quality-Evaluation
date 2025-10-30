class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
            
        n = len(s)
        special_substrings = []
        
        # Find all special substrings
        for i in range(n):
            for j in range(i + 1, n):  # j is exclusive, so substring is s[i:j]
                if j - i == n:  # Skip if it's the entire string
                    continue
                    
                substring = s[i:j]
                is_special = True
                
                # Check if any character in substring appears outside [i, j)
                for char in set(substring):
                    # Check before position i
                    if char in s[:i]:
                        is_special = False
                        break
                    # Check after position j-1
                    if char in s[j:]:
                        is_special = False
                        break
                
                if is_special:
                    special_substrings.append((i, j))
        
        # Now find maximum number of disjoint substrings
        if not special_substrings:
            return False
            
        # Sort by end position for greedy selection
        special_substrings.sort(key=lambda x: x[1])
        
        count = 0
        last_end = -1
        
        for start, end in special_substrings:
            if start >= last_end:  # Non-overlapping
                count += 1
                last_end = end
        
        return count >= k