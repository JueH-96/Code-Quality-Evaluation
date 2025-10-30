class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        total_length = n + m - 1
        word = ['?'] * total_length
        
        # First process all 'T' constraints to set necessary characters
        for i in range(n):
            if str1[i] == 'T':
                # The substring word[i..i+m-1] must be str2
                for k in range(m):
                    pos = i + k
                    if pos >= total_length:
                        return ""  # impossible, since i + m -1 must be < total_length
                    if word[pos] == '?':
                        word[pos] = str2[k]
                    elif word[pos] != str2[k]:
                        return ""  # conflict
        
        # Now process 'F' constraints
        for i in range(n):
            if str1[i] == 'F':
                # The substring must not be str2
                # Check if current constraints already make it impossible
                # Collect the current substring
                substring_positions = range(i, i + m)
                if i + m > total_length:
                    return ""  # invalid, since i + m -1 >= total_length
                # Check if the current word matches str2 exactly in all set positions
                match = True
                for k in range(m):
                    pos = i + k
                    if word[pos] != '?' and word[pos] != str2[k]:
                        match = False
                        break
                if match:
                    # Need to change at least one character in the substring
                    changed = False
                    for k in range(m):
                        pos = i + k
                        if word[pos] == '?':
                            # We can set this to something other than str2[k]
                            # Choose the smallest possible character that is not str2[k]
                            for c in 'abcdefghijklmnopqrstuvwxyz':
                                if c != str2[k]:
                                    word[pos] = c
                                    changed = True
                                    break
                            if changed:
                                break
                    if not changed:
                        return ""  # no position to change, all are set to match str2
        
        # Fill remaining '?' with 'a's
        for i in range(total_length):
            if word[i] == '?':
                word[i] = 'a'
        
        # Final check for all 'F' constraints
        s = ''.join(word)
        for i in range(n):
            if i + m > total_length:
                continue  # shouldn't happen per problem constraints
            substring = s[i:i+m]
            if str1[i] == 'T':
                if substring != str2:
                    return ""
            else:
                if substring == str2:
                    return ""
        
        return s