class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        str1 = str1 + str1
        ptr = 0
        for ch in str2:
            pos = str1.find(ch, ptr)
            if pos == -1:
                pos = str1.find(ch, 0)
                if pos == -1 or pos < ptr:
                    return False
            ptr = pos + 1
        return True