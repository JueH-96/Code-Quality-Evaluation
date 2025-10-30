class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def swap(n):
            res = []
            s = str(n)
            for i in range(len(s)):
                for j in range(len(s)):
                    if i != j:
                        temp = list(s)
                        temp[i], temp[j] = temp[j], temp[i]
                        res.append(int("".join(temp)))
            return res

        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j] or nums[i] in swap(nums[j]) or nums[j] in swap(nums[i]):
                    count += 1
        return count