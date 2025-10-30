class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        from collections import Counter
        n = len(s)
        result = []
        
        for a, b, c, d in queries:
            # Characters available for rearrangement
            left_chars = Counter(s[a:b+1])
            right_chars = Counter(s[c:d+1])
            
            can_make = True
            
            # Check each palindrome constraint
            for i in range(n // 2):
                j = n - 1 - i  # Mirror position
                
                left_modifiable = a <= i <= b
                right_modifiable = c <= j <= d
                
                if left_modifiable and right_modifiable:
                    # Both positions can be changed, handle at the end
                    continue
                elif left_modifiable:
                    # Left can change, right is fixed
                    needed_char = s[j]
                    if left_chars[needed_char] > 0:
                        left_chars[needed_char] -= 1
                    else:
                        can_make = False
                        break
                elif right_modifiable:
                    # Right can change, left is fixed
                    needed_char = s[i]
                    if right_chars[needed_char] > 0:
                        right_chars[needed_char] -= 1
                    else:
                        can_make = False
                        break
                else:
                    # Both are fixed, must already match
                    if s[i] != s[j]:
                        can_make = False
                        break
            
            if can_make:
                # Check if remaining characters can form matching pairs
                left_remaining = {k: v for k, v in left_chars.items() if v > 0}
                right_remaining = {k: v for k, v in right_chars.items() if v > 0}
                result.append(left_remaining == right_remaining)
            else:
                result.append(False)
        
        return result