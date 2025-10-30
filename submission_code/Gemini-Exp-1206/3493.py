class Solution:
    def maxOperations(self, s: str) -> int:
        ones = 0
        zeros = 0
        ans = 0
        for char in s:
            if char == '1':
                ones += 1
            else:
                if ones > 0:
                    ans += 1
                    ones -= 1
                    zeros += 1
                else:
                    zeros +=1
        return ans