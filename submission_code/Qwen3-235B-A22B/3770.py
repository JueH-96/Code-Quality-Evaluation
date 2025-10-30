class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        len_word = n + m - 1
        if m == 0:
            return ""
        
        res = [None] * len_word

        # Step 1: Apply all 'T' constraints and check for conflicts
        for i in range(n):
            if str1[i] == 'T':
                for k in range(m):
                    pos = i + k
                    if pos >= len_word:
                        return ""
                    if res[pos] is None:
                        res[pos] = str2[k]
                    else:
                        if res[pos] != str2[k]:
                            return ""
        
        # Step 2: Check all 'F' constraints to ensure they can be satisfied
        for i in range(n):
            if str1[i] == 'F':
                can_diff = False
                for k in range(m):
                    pos = i + k
                    if pos >= len_word:
                        can_diff = True
                        break
                    if res[pos] is None:
                        can_diff = True
                        break
                    elif res[pos] != str2[k]:
                        can_diff = True
                        break
                if not can_diff:
                    return ""
        
        # Step 3: Fill all remaining positions with 'a'
        for j in range(len_word):
            if res[j] is None:
                res[j] = 'a'
        
        # Step 4: Collect all F windows that now match and need adjustment
        invalid_fs = []
        for i in range(n):
            if str1[i] == 'F':
                match = True
                for k in range(m):
                    pos = i + k
                    if pos >= len_word:
                        match = False
                        break
                    if res[pos] != str2[k]:
                        match = False
                        break
                if match:
                    invalid_fs.append(i)
        
        if not invalid_fs:
            return ''.join(res)
        
        # Step 5: Adjust the earliest possible position in the earliest invalid window
        # Find the earliest j in all invalid_fs windows
        found = False
        for j in range(len_word):
            # Check if j is in any of the invalid_fs windows
            for i in invalid_fs:
                if i <= j < i + m:
                    offset = j - i
                    # We need to change res[j] to break the match
                    original = res[j]
                    # Find the smallest possible character different from str2[offset]
                    if str2[offset] < 'z':
                        new_char = chr(ord(str2[offset]) + 1)
                        if new_char == 'z' + 1:
                            new_char = 'z'
                    else:
                        new_char = 'y'
                    
                    if str2[offset] == 'a':
                        candidate = 'a'
                    else:
                        candidate = 'a'
                    if candidate != str2[offset]:
                        new_char = candidate
                    else:
                        new_char = 'b'
                    
                    # Choose the smaller new_char
                    if new_char != str2[offset]:
                        pass
                    else:
                        new_char = chr(ord(new_char) + 1)
                    
                    if new_char < res[j]:
                        # Change to new_char
                        l_res = list(res)
                        l_res[j] = new_char
                        res = l_res
                        # Re-check all invalid_fs
                        new_invalid = []
                        for ii in invalid_fs:
                            match = True
                            for k in range(m):
                                pos = ii + k
                                if pos >= len_word:
                                    match = False
                                    break
                                if res[pos] != str2[k]:
                                    match = False
                                    break
                            if match:
                                new_invalid.append(ii)
                        invalid_fs = new_invalid
                        if not invalid_fs:
                            return ''.join(res)
                        # Restart the loop since invalid_fs has been updated
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        # After adjustment, check if invalid_fs remains
        if invalid_fs:
            return ""
        return ''.join(res)