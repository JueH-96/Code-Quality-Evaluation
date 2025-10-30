class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        fixed = [None] * L
        
        # Step 1: Process all 'T's and check for conflicts
        for i in range(n):
            if str1[i] == 'T':
                for j in range(m):
                    pos = i + j
                    if fixed[pos] is not None:
                        if fixed[pos] != str2[j]:
                            return ""
                    else:
                        fixed[pos] = str2[j]
        
        # Step 2: Fill the candidate with 'a's where not fixed
        candidate = []
        for c in fixed:
            if c is not None:
                candidate.append(c)
            else:
                candidate.append('a')
        
        # Step 3: Process all 'F's
        for i in range(n):
            if str1[i] == 'F':
                # Check if the current substring equals str2
                substr = []
                for j in range(m):
                    pos = i + j
                    substr.append(candidate[pos])
                if ''.join(substr) == str2:
                    modified = False
                    for j in range(m):
                        pos = i + j
                        if fixed[pos] is None:  # This position can be modified
                            target_char = str2[j]
                            # Find the smallest character different from target_char
                            for c in 'abcdefghijklmnopqrstuvwxyz':
                                if c != target_char:
                                    candidate[pos] = c
                                    modified = True
                                    break
                            if modified:
                                break
                    if not modified:
                        return ""
        
        return ''.join(candidate)