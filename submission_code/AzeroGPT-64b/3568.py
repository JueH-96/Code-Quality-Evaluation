class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        a=str(num1)
        b=str(num2)
        c=str(num3)
        ans=''
        for i in range(4):
            min_digit=min(int((a if len(a) > i else "0"*(4-len(a)) + a)[i]),
                          int((b if len(b) > i else "0"*(4-len(b)) + b)[i]),
                          int((c if len(c) > i else "0"*(4-len(c)) + c)[i]))
            ans+=str(min_digit)
        return int(ans.lstrip('0'))