class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        def canFormPalindrome(a, b, c, d):
            # Convert to lists for easier manipulation
            left_part = list(s[:half])
            right_part = list(s[half:])
            
            # Get the rearrangeable portions
            left_rearrange = left_part[a:b+1]
            right_rearrange = right_part[c-half:d-half+1]
            
            # Check positions that cannot be rearranged
            for i in range(half):
                mirror_pos = half - 1 - i
                
                # If both positions are outside rearrangeable ranges
                if not (a <= i <= b) and not (c-half <= mirror_pos <= d-half):
                    if left_part[i] != right_part[mirror_pos]:
                        return False
                        
            # Count characters in rearrangeable portions
            from collections import Counter
            left_counter = Counter(left_rearrange)
            right_counter = Counter(right_rearrange)
            
            # For positions where one side is rearrangeable and other is not
            for i in range(half):
                mirror_pos = half - 1 - i
                
                if (a <= i <= b) and not (c-half <= mirror_pos <= d-half):
                    # Left side rearrangeable, right side fixed
                    needed_char = right_part[mirror_pos]
                    if left_counter[needed_char] > 0:
                        left_counter[needed_char] -= 1
                    else:
                        return False
                        
                elif not (a <= i <= b) and (c-half <= mirror_pos <= d-half):
                    # Right side rearrangeable, left side fixed
                    needed_char = left_part[i]
                    if right_counter[needed_char] > 0:
                        right_counter[needed_char] -= 1
                    else:
                        return False
            
            # For positions where both sides are rearrangeable
            # We need to check if remaining characters can form palindrome pairs
            total_left = sum(left_counter.values())
            total_right = sum(right_counter.values())
            
            if total_left != total_right:
                return False
                
            # Check if we can pair up the remaining characters
            all_chars = set(left_counter.keys()) | set(right_counter.keys())
            for char in all_chars:
                left_count = left_counter[char]
                right_count = right_counter[char]
                # The difference should be even (so we can split evenly)
                if (left_count + right_count) % 2 != 0:
                    return False
                    
            return True
        
        result = []
        for query in queries:
            a, b, c, d = query
            result.append(canFormPalindrome(a, b, c, d))
            
        return result