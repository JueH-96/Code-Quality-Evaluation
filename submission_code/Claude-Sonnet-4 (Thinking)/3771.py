class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Find leftmost and rightmost occurrence of each character
        left = {}
        right = {}
        for i, c in enumerate(s):
            if c not in left:
                left[c] = i
            right[c] = i
        
        # Find all special substrings
        special_substrings = []
        
        for i in range(n):
            for j in range(i + 1, n + 1):  # j is exclusive, so substring is s[i:j]
                if i == 0 and j == n:  # Skip the entire string
                    continue
                
                # Check if this substring is special
                is_special = True
                chars_in_substring = set(s[i:j])
                for c in chars_in_substring:
                    if left[c] < i or right[c] >= j:
                        is_special = False
                        break
                
                if is_special:
                    special_substrings.append((i, j))
        
        # Find the maximum number of disjoint special substrings using greedy algorithm
        special_substrings.sort(key=lambda x: x[1])
        
        count = 0
        last_end = -1
        
        for start, end in special_substrings:
            if start >= last_end:
                count += 1
                last_end = end
        
        return count >= k