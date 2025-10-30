class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # Precompute prefix counts for both halves
        prefix_left = [[0] * 26]
        prefix_right = [[0] * 26]
        
        # Build prefix counts for left half
        for i in range(half):
            curr = prefix_left[-1].copy()
            curr[ord(s[i]) - ord('a')] += 1
            prefix_left.append(curr)
            
        # Build prefix counts for right half (reversed)
        for i in range(n-1, half-1, -1):
            curr = prefix_right[-1].copy()
            curr[ord(s[i]) - ord('a')] += 1
            prefix_right.append(curr)
            
        def get_freq(prefix, start, end):
            freq = [0] * 26
            for i in range(26):
                freq[i] = prefix[end+1][i] - prefix[start][i]
            return freq
            
        def check_equal(freq1, freq2):
            return all(f1 == f2 for f1, f2 in zip(freq1, freq2))
            
        def check_palindrome(start, end):
            left_freq = get_freq(prefix_left, start, end)
            right_freq = get_freq(prefix_right, n-1-end, n-1-start)
            return check_equal(left_freq, right_freq)
            
        result = []
        for a, b, c, d in queries:
            # Check if parts outside query ranges are palindromes
            is_valid = True
            
            # Check left side before a
            if a > 0 and not check_palindrome(0, a-1):
                is_valid = False
                
            # Check middle part between b and c
            if b+1 < half and c-1 >= half:
                if not check_palindrome(b+1, half-1):
                    is_valid = False
                if not check_palindrome(half, c-1):
                    is_valid = False
                    
            # Check right side after d
            if d < n-1 and not check_palindrome(d+1, n-1):
                is_valid = False
                
            # Check if characters in query ranges can be rearranged to form palindrome
            if is_valid:
                # Get all characters in left and right query ranges
                left_freq = get_freq(prefix_left, a, b)
                right_freq = get_freq(prefix_right, n-1-d, n-1-c)
                is_valid = check_equal(left_freq, right_freq)
                
            result.append(is_valid)
            
        return result