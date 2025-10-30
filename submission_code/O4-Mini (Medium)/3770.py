class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        
        # word array, '?' = unknown
        word = ['?'] * L
        # forced positions by 'T'
        forced = [False] * L
        
        # 1) Apply all 'T' constraints
        for i, tf in enumerate(str1):
            if tf == 'T':
                # substring word[i:i+m] must equal str2
                for k in range(m):
                    j = i + k
                    c = str2[k]
                    if word[j] == '?' or word[j] == c:
                        word[j] = c
                        forced[j] = True
                    else:
                        # conflict
                        return ""
        
        # 2) Fill remaining unknowns with 'a' for minimal lex
        for i in range(L):
            if word[i] == '?':
                word[i] = 'a'
        
        # 3) Enforce all 'F' constraints by making sure
        #    word[i:i+m] != str2 for each 'F' in str1
        for i, tf in enumerate(str1):
            if tf == 'F':
                # check if substring equals str2
                match = True
                for k in range(m):
                    if word[i + k] != str2[k]:
                        match = False
                        break
                if match:
                    # need to break equality by changing some free position
                    broken = False
                    # pick rightmost free position in this window
                    for k in range(m - 1, -1, -1):
                        j = i + k
                        if not forced[j]:
                            # choose the smallest character != str2[k]
                            if str2[k] != 'a':
                                word[j] = 'a'
                            else:
                                word[j] = 'b'
                            broken = True
                            break
                    if not broken:
                        # no way to break this forbidden match
                        return ""
        
        return "".join(word)