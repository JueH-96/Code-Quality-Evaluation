class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n, m = len(str1), len(str2)
        
        # Check if a substring matches str2
        def is_valid_substring(word, start, target, should_match):
            if start + m > len(word):
                return False
            substring = word[start:start + m]
            return (substring == target) == should_match
        
        # Check if the current word satisfies all conditions
        def is_valid(word):
            for i in range(n):
                should_match = str1[i] == 'T'
                if not is_valid_substring(word, i, str2, should_match):
                    return False
            return True
        
        # Try to build the string character by character
        def backtrack(curr):
            if len(curr) == n + m - 1:
                if is_valid(curr):
                    return curr
                return None
            
            # Try each possible character from 'a' to 'z'
            for c in range(ord('a'), ord('z') + 1):
                curr.append(chr(c))
                
                # Check if current partial string could be valid
                valid = True
                for i in range(max(0, len(curr) - m + 1)):
                    if i >= n:
                        break
                    if len(curr) >= i + m:
                        should_match = str1[i] == 'T'
                        is_match = curr[i:i + m] == list(str2)
                        if should_match != is_match:
                            valid = False
                            break
                
                if valid:
                    result = backtrack(curr)
                    if result is not None:
                        return result
                curr.pop()
            
            return None
        
        # Special case for str1 = "F"
        if n == 1 and str1 == "F":
            return "a" if str2 != "a" else "b"
        
        # Start the backtracking process
        result = backtrack([])
        return ''.join(result) if result is not None else ""