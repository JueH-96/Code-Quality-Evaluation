class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        result_len = n + m - 1
        
        # Initialize result with None values
        result = [None] * result_len
        
        # For each position, try to find the smallest valid character
        for pos in range(result_len):
            # Try each character from 'a' to 'z'
            for char in 'abcdefghijklmnopqrstuvwxyz':
                result[pos] = char
                
                # Check if this assignment is valid
                valid = True
                
                # Check all constraints that involve position pos
                for i in range(n):
                    # Check if position pos is part of substring starting at i
                    if i <= pos < i + m:
                        # Build the substring starting at position i
                        substring = []
                        all_assigned = True
                        
                        for j in range(i, i + m):
                            if result[j] is None:
                                all_assigned = False
                                break
                            substring.append(result[j])
                        
                        # Only check if all positions in substring are assigned
                        if all_assigned:
                            substring_str = ''.join(substring)
                            
                            if str1[i] == 'T' and substring_str != str2:
                                valid = False
                                break
                            elif str1[i] == 'F' and substring_str == str2:
                                valid = False
                                break
                
                if valid:
                    # Keep this character and move to next position
                    break
            else:
                # No valid character found for this position
                return ""
        
        return ''.join(result)