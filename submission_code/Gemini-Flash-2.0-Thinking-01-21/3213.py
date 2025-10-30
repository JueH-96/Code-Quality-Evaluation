class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_element = 0
        for num in nums:
            max_element = max(max_element, num)

        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                max_element_count = 0
                for element in sub_array:
                    if element == max_element:
                        max_element_count += 1
                if max_element_count >= k:
                    count += 1
        return count