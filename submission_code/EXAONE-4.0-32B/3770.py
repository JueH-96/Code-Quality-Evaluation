class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1
        ans = [''] * L
        current_match = [True] * n
        
        for j in range(L):
            low = max(0, j - m + 1)
            high = min(j, n - 1)
            if low > high:
                ans[j] = 'a'
                continue
            
            forced_char = None
            for i in range(low, high + 1):
                if str1[i] == 'T':
                    k = j - i
                    c_req = str2[k]
                    if forced_char is None:
                        forced_char = c_req
                    elif forced_char != c_req:
                        return ""
            
            if forced_char is not None:
                c = forced_char
                for i in range(low, high + 1):
                    k = j - i
                    if c != str2[k]:
                        current_match[i] = False
                i0 = j - m + 1
                if i0 >= 0 and i0 < n:
                    if str1[i0] == 'T':
                        if not current_match[i0]:
                            return ""
                    else:
                        if current_match[i0]:
                            return ""
                ans[j] = c
            else:
                found = False
                for candidate in "abcdefghijklmnopqrstuvwxyz":
                    backup = []
                    for i in range(low, high + 1):
                        backup.append((i, current_match[i]))
                    for i in range(low, high + 1):
                        k = j - i
                        if candidate != str2[k]:
                            current_match[i] = False
                    i0 = j - m + 1
                    valid = True
                    if i0 >= 0 and i0 < n:
                        if str1[i0] == 'T':
                            if not current_match[i0]:
                                valid = False
                        else:
                            if current_match[i0]:
                                valid = False
                    if valid:
                        c = candidate
                        found = True
                        break
                    else:
                        for (i, val) in backup:
                            current_match[i] = val
                if not found:
                    return ""
                ans[j] = c
        
        return ''.join(ans)