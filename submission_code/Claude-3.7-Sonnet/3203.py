class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        results = []

        for a, b, c, d in queries:
            # Check characters outside both rearrangeable regions for palindrome property
            is_palindrome = True
            
            for i in range(n // 2):
                corresponding_idx = n - 1 - i
                
                if (i < a or i > b) and (corresponding_idx < c or corresponding_idx > d):
                    if s[i] != s[corresponding_idx]:
                        is_palindrome = False
                        break
            
            if not is_palindrome:
                results.append(False)
                continue
            
            # Create character frequency maps for both rearrangeable regions
            char_count_first = {}
            for i in range(a, b+1):
                char_count_first[s[i]] = char_count_first.get(s[i], 0) + 1
            
            char_count_second = {}
            for i in range(c, d+1):
                char_count_second[s[i]] = char_count_second.get(s[i], 0) + 1
            
            # Check for positions in the first rearrangeable region that need to match fixed characters
            for i in range(a, b+1):
                corresponding_idx = n - 1 - i
                if corresponding_idx < c or corresponding_idx > d:
                    char = s[corresponding_idx]
                    if char_count_first.get(char, 0) > 0:
                        char_count_first[char] -= 1
                        if char_count_first[char] == 0:
                            del char_count_first[char]
                    else:
                        is_palindrome = False
                        break
            
            if not is_palindrome:
                results.append(False)
                continue
            
            # Check for positions in the second rearrangeable region that need to match fixed characters
            for i in range(c, d+1):
                corresponding_idx = n - 1 - i
                if corresponding_idx < a or corresponding_idx > b:
                    char = s[corresponding_idx]
                    if char_count_second.get(char, 0) > 0:
                        char_count_second[char] -= 1
                        if char_count_second[char] == 0:
                            del char_count_second[char]
                    else:
                        is_palindrome = False
                        break
            
            if not is_palindrome:
                results.append(False)
                continue
            
            # Check if the remaining characters in both rearrangeable regions match
            if char_count_first != char_count_second:
                is_palindrome = False
            
            results.append(is_palindrome)
        
        return results