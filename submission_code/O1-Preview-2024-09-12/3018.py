class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0
        j = 0
        performed_operation = False
        n1 = len(str1)
        n2 = len(str2)
        ord_a = ord('a')
        
        while i < n1 and j < n2:
            ch1 = str1[i]
            ch2 = str2[j]
            if ch1 == ch2:
                i += 1
                j += 1
            else:
                ch1_inc = chr((ord(ch1) - ord_a + 1)%26 + ord_a)
                if ch1_inc == ch2:
                    # We can increment this character
                    i += 1
                    j += 1
                else:
                    i +=1
        return j == n2