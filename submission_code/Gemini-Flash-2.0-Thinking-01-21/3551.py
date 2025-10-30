from typing import List

class Solution:
    def calculate_xor_score(self, arr: List[int]) -> int:
        current_arr = list(arr)
        if not current_arr:
            return 0
        while len(current_arr) > 1:
            next_arr = []
            for i in range(len(current_arr) - 1):
                next_arr.append(current_arr[i] ^ current_arr[i+1])
            current_arr = next_arr
        return current_arr[0] if current_arr else 0
        
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        results = []
        for query in queries:
            l_i, r_i = query[0], query[1]
            max_xor_score = 0
            for start_index in range(l_i, r_i + 1):
                for end_index in range(start_index, r_i + 1):
                    subarray = nums[start_index : end_index + 1]
                    current_score = self.calculate_xor_score(subarray)
                    max_xor_score = max(max_xor_score, current_score)
            results.append(max_xor_score)
        return results