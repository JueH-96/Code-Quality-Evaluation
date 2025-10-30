class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Edge case
        if k == 0:
            return True
        
        # Find first and last occurrence of each character
        first_occurrence = {}
        last_occurrence = {}
        for i, char in enumerate(s):
            if char not in first_occurrence:
                first_occurrence[char] = i
            last_occurrence[char] = i
        
        # Find all special substrings
        special_substrings = []
        for start in range(n):
            for end in range(start, n - 1):  # end < n - 1 to ensure it's not the entire string
                is_special = True
                # Check if all characters in substring have all their occurrences within it
                for i in range(start, end + 1):
                    char = s[i]
                    if first_occurrence[char] < start or last_occurrence[char] > end:
                        is_special = False
                        break
                if is_special:
                    special_substrings.append((start, end))
        
        # Find maximum number of disjoint special substrings using greedy approach
        # Sort by end position for optimal interval scheduling
        special_substrings.sort(key=lambda x: x[1])
        count = 0
        last_end = -1
        for start, end in special_substrings:
            if start > last_end:
                count += 1
                last_end = end
        
        return count >= k