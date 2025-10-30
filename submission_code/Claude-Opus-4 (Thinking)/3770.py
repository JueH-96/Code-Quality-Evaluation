class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        length = n + m - 1
        
        # Step 1: Identify fixed positions from 'T' constraints
        fixed = {}
        for i in range(n):
            if str1[i] == 'T':
                for j in range(m):
                    pos = i + j
                    if pos in fixed and fixed[pos] != str2[j]:
                        return ""  # Conflicting constraints
                    fixed[pos] = str2[j]
        
        # Step 2: Build string character by character
        result = []
        for pos in range(length):
            if pos in fixed:
                result.append(fixed[pos])
            else:
                # Try each character in lexicographical order
                found = False
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    # Check if this character violates any constraint
                    valid = True
                    
                    # Check all constraints that end at current position
                    for i in range(max(0, pos - m + 1), min(pos + 1, n)):
                        if i + m - 1 == pos:  # This constraint can be fully checked
                            # Build the substring for this constraint
                            substring = []
                            for k in range(m):
                                if i + k < len(result):
                                    substring.append(result[i + k])
                                else:
                                    substring.append(c)
                            
                            matches = all(substring[j] == str2[j] for j in range(m))
                            
                            if str1[i] == 'T' and not matches:
                                valid = False
                                break
                            if str1[i] == 'F' and matches:
                                valid = False
                                break
                    
                    if valid:
                        result.append(c)
                        found = True
                        break
                
                if not found:
                    return ""
        
        return ''.join(result)