class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            if num % 2 == 0:
                ans.append(-1)
            elif (num + 1) & num == 0 and (num + 1) > 0:
                res = (num + 1) // 2 - 1
                if (res | (res + 1)) == num:
                    ans.append(res)
                else:
                    ans.append(-1)
            else:
                found = False
                for j in range(num):
                    if (j | (j + 1)) == num:
                        ans.append(j)
                        found = True
                        break
                if not found:
                    ans.append(-1)
        return ans