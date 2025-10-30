class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        from collections import Counter
        
        n = len(s)
        half_n = n // 2
        
        # Count frequency of characters in the first half and second half
        left_count = Counter(s[:half_n])
        right_count = Counter(s[half_n:])
        
        results = []
        
        for a, b, c, d in queries:
            # Create a new counter for the left and right parts after rearrangement
            left_part = Counter(s[a:b + 1])
            right_part = Counter(s[c:d + 1])
            
            # Update the counts
            for char in left_part:
                left_count[char] += left_part[char]
            for char in right_part:
                right_count[char] += right_part[char]
            
            # Check if we can form a palindrome
            odd_count = 0
            
            # Check the combined counts for both halves
            for char in set(left_count.keys()).union(right_count.keys()):
                if (left_count[char] + right_count[char]) % 2 != 0:
                    odd_count += 1
            
            # A palindrome can have at most one character with an odd count
            results.append(odd_count <= 1)
            
            # Revert the counts back to original for the next query
            for char in left_part:
                left_count[char] -= left_part[char]
            for char in right_part:
                right_count[char] -= right_part[char]
        
        return results