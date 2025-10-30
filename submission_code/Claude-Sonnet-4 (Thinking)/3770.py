class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n, m = len(str1), len(str2)
        result = [None] * (n + m - 1)
        
        # Apply 'T' constraints: force substrings to equal str2
        for i in range(n):
            if str1[i] == 'T':
                for j in range(m):
                    if result[i + j] is None:
                        result[i + j] = str2[j]
                    elif result[i + j] != str2[j]:
                        return ""  # Contradiction
        
        # Apply 'F' constraints: ensure substrings don't equal str2
        for i in range(n):
            if str1[i] == 'F':
                # Check if substring could equal str2 with current assignment
                could_equal = True
                for j in range(m):
                    if result[i + j] is not None and result[i + j] != str2[j]:
                        could_equal = False
                        break
                
                if could_equal:
                    # Try setting all unset positions to 'a'
                    temp_substring = []
                    for j in range(m):
                        if result[i + j] is not None:
                            temp_substring.append(result[i + j])
                        else:
                            temp_substring.append('a')
                    
                    if ''.join(temp_substring) != str2:
                        # Setting unset positions to 'a' works
                        for j in range(m):
                            if result[i + j] is None:
                                result[i + j] = 'a'
                    else:
                        # Need to set one position differently
                        # Choose rightmost unset position for minimal lexicographical impact
                        fixed = False
                        for j in range(m - 1, -1, -1):
                            if result[i + j] is None:
                                result[i + j] = 'a' if str2[j] != 'a' else 'b'
                                fixed = True
                                break
                        
                        if not fixed:
                            return ""  # All positions set and equal to str2
        
        # Fill remaining positions with 'a'
        for i in range(len(result)):
            if result[i] is None:
                result[i] = 'a'
        
        return ''.join(result)