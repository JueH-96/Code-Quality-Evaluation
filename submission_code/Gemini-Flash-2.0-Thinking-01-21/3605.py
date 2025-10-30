class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans_list = []
        for num in nums:
            if num == 2:
                ans_list.append(-1)
            else:
                found = False
                for a in range(num):
                    if (a | (a + 1)) == num:
                        ans_list.append(a)
                        found = True
                        break
                if not found:
                    ans_list.append(-1)
        return ans_list