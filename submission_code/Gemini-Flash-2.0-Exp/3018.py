class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n1 = len(str1)
        n2 = len(str2)
        
        def is_subsequence(s1, s2):
            i = 0
            j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    j += 1
                i += 1
            return j == len(s2)
        
        if is_subsequence(str1, str2):
            return True
        
        for i in range(n1):
            temp_str1 = list(str1)
            temp_str1[i] = chr(((ord(str1[i]) - ord('a') + 1) % 26) + ord('a'))
            temp_str1 = "".join(temp_str1)
            
            if is_subsequence(temp_str1, str2):
                return True
        
        
        
        temp_str1 = list(str1)
        changed_indices = []
        
        def solve():
            nonlocal temp_str1
            nonlocal changed_indices
            
            if is_subsequence("".join(temp_str1), str2):
                return True
            
            if len(changed_indices) > 0:
                return False
            
            for i in range(n1):
                original_char = temp_str1[i]
                temp_str1[i] = chr(((ord(str1[i]) - ord('a') + 1) % 26) + ord('a'))
                changed_indices.append(i)
                
                if is_subsequence("".join(temp_str1), str2):
                    return True
                
                changed_indices.pop()
                temp_str1[i] = original_char
                
            return False
        
        return solve()