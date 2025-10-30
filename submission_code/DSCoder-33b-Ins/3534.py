class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def swap(num):
            num = str(num)
            res = []
            for i in range(len(num)):
                for j in range(i+1, len(num)):
                    num_list = list(num)
                    num_list[i], num_list[j] = num_list[j], num_list[i]
                    if num_list[0] != '0':
                        res.append(int(''.join(num_list)))
            return res

        count = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] in swap(nums[j]) or nums[j] in swap(nums[i]):
                    count += 1
        return count