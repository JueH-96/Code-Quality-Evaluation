from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        def imbalance_of_subarray(subarr):
            sarr = sorted(subarr)
            imbalance = 0
            for i in range(len(sarr) - 1):
                if sarr[i + 1] - sarr[i] > 1:
                    imbalance += 1
            return imbalance
        
        total_imbalance = 0
        n = len(nums)
        
        for start in range(n):
            unique_elements = set()
            current_imbalance = 0
            for end in range(start, n):
                if nums[end] not in unique_elements:
                    unique_elements.add(nums[end])
                    sorted_insert_pos = len(unique_elements)
                    for i, num in enumerate(sorted(unique_elements)):
                        if num > nums[end]:
                            sorted_insert_pos = i
                            break
                    if sorted_insert_pos > 0 and nums[end] - sorted(unique_elements)[sorted_insert_pos - 1] > 1:
                        current_imbalance += 1
                    if sorted_insert_pos < len(unique_elements) - 1 and sorted(unique_elements)[sorted_insert_pos + 1] - nums[end] > 1:
                        current_imbalance += 1
                    if sorted_insert_pos > 0 and sorted_insert_pos < len(unique_elements) - 1:
                        if sorted(unique_elements)[sorted_insert_pos + 1] - sorted(unique_elements)[sorted_insert_pos - 1] <= 1:
                            current_imbalance -= 1
                total_imbalance += current_imbalance
        
        return total_imbalance