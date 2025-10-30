class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        # Step 1: Process T constraints and build fixed array
        fixed = [None] * L
        for i in range(n):
            if str1[i] == 'T':
                for k in range(m):
                    pos = i + k
                    if fixed[pos] is None:
                        fixed[pos] = str2[k]
                    else:
                        if fixed[pos] != str2[k]:
                            return ""
        
        # Step 2: Check F constraints for fully fixed substrings
        for i in range(n):
            if str1[i] == 'F':
                all_fixed = True
                equals_str2 = True
                for k in range(m):
                    pos = i + k
                    if fixed[pos] is None:
                        all_fixed = False
                        break
                    if fixed[pos] != str2[k]:
                        equals_str2 = False
                if all_fixed and equals_str2:
                    return ""
        
        # Step 3: Build initial word with fixed and 'a's
        word = []
        for pos in range(L):
            if fixed[pos] is not None:
                word.append(fixed[pos])
            else:
                word.append('a')
        
        # Step 4: Process each F to ensure substrings don't match str2
        for i in range(n):
            if str1[i] == 'F':
                # Check if the substring matches str2
                match = True
                for k in range(m):
                    pos = i + k
                    if word[pos] != str2[k]:
                        match = False
                        break
                if not match:
                    continue  # substring is already different
                
                # Need to change: find the rightmost position to alter
                changed = False
                for j in range(i + m - 1, i - 1, -1):
                    k_val = j - i
                    target_char = str2[k_val]
                    if word[j] == target_char and fixed[j] is None:
                        # Find the smallest possible different character
                        new_char = 'a'
                        while new_char == target_char:
                            new_char = chr(ord(new_char) + 1)
                        word[j] = new_char
                        changed = True
                        break
                if not changed:
                    return ""  # This should not happen due to earlier checks
        
        return ''.join(word)