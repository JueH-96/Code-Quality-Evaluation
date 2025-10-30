from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Create a sorted list of tuples (value, original index)
        sorted_list = sorted([(nums[i], i) for i in range(len(nums))], key=lambda x: (x[0], x[1]))
        # Create a dictionary to map original index to its position in the sorted list
        pos_in_sorted = {i: idx for idx, (v, i) in enumerate(sorted_list)}
        marked = [False] * len(nums)
        sum_marked = 0
        total_sum = sum(nums)
        ptr = 0  # Pointer to track the current position in the sorted list
        answer = []
        
        for index_i, k_i in queries:
            # Step 1: Mark the element at index_i if it's not already marked
            if not marked[index_i]:
                sum_marked += nums[index_i]
                marked[index_i] = True
            
            # Step 2: Mark up to k_i smallest unmarked elements
            count = 0
            while count < k_i and ptr < len(sorted_list):
                original_index = sorted_list[ptr][1]
                if not marked[original_index]:
                    sum_marked += nums[original_index]
                    marked[original_index] = True
                    count += 1
                ptr += 1  # Move to the next element regardless of whether it was marked
            
            # Append the current sum of unmarked elements
            answer.append(total_sum - sum_marked)
        
        return answer