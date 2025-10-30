class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        if n == 0 or k == 0:
            return 0
        
        # Precompute the adjacency validity array
        adj = [False] * (n - 1)
        for i in range(n - 1):
            c1 = ord(word[i]) - ord('a')
            c2 = ord(word[i + 1]) - ord('a')
            if abs(c1 - c2) <= 2:
                adj[i] = True
        
        # Precompute the prefix sum of valid adjacency pairs
        prefix = [0] * (n)
        for i in range(1, n):
            prefix[i] = prefix[i - 1] + (1 if i - 1 < len(adj) and adj[i - 1] else 0)
        
        left = 0
        freq = {}
        valid_chars = 0
        distinct_chars = 0
        result = 0
        
        for right in range(n):
            char = word[right]
            if freq.get(char, 0) == 0:
                distinct_chars += 1
            freq[char] = freq.get(char, 0) + 1
            
            # Ensure no character exceeds k occurrences
            while freq[char] > k:
                left_char = word[left]
                freq[left_char] -= 1
                if freq[left_char] == k:
                    valid_chars -= 1
                if freq[left_char] == 0:
                    distinct_chars -= 1
                left += 1
            
            # Update valid_chars if current character's count is exactly k
            if freq[char] == k:
                valid_chars += 1
            
            # Check if current window meets the conditions
            m = distinct_chars
            if m == 0:
                continue
            if valid_chars == m and (right - left + 1) == m * k:
                # Check adjacency condition using prefix sum
                if prefix[right] - prefix[left] == (right - left):
                    result += 1
        
        return result