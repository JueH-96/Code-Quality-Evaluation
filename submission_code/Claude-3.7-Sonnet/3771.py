class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        # If k is 0, we don't need any special substrings
        if k == 0:
            return True
        
        n = len(s)
        visited = [False] * n  # Track which positions are already part of a special substring
        special_count = 0  # Count of special substrings found
        
        for i in range(n):
            if visited[i]:
                continue
            
            for j in range(i + 1, n + 1):
                # Check if s[i:j] is a special substring
                if self.is_special(s, i, j, visited):
                    # Mark all positions in this substring as visited
                    for idx in range(i, j):
                        visited[idx] = True
                    special_count += 1
                    break  # Found a special substring starting at i, move to next unvisited position
            
            if special_count >= k:
                return True
        
        return special_count >= k
    
    def is_special(self, s: str, start: int, end: int, visited: list[bool]) -> bool:
        # Check if this is the entire string
        if start == 0 and end == len(s):
            return False
        
        # Check if any part of substring is already visited
        for idx in range(start, end):
            if visited[idx]:
                return False
        
        # Check if characters in the substring appear elsewhere
        chars_inside = set(s[start:end])
        for char in chars_inside:
            for idx in range(len(s)):
                # Skip if within the substring or already visited
                if idx >= start and idx < end or visited[idx]:
                    continue
                
                # If character appears outside the substring, it's not special
                if s[idx] == char:
                    return False
        
        return True