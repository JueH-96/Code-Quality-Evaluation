class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def calculate_xor_score(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0]
            
            while len(arr) > 1:
                new_arr = []
                for i in range(len(arr) - 1):
                    new_arr.append(arr[i] ^ arr[i+1])
                arr = new_arr
            return arr[0]

        def find_max_xor_score(sub_array):
            max_xor = 0
            for i in range(len(sub_array)):
                for j in range(i, len(sub_array)):
                    current_sub_array = sub_array[i:j+1]
                    max_xor = max(max_xor, calculate_xor_score(current_sub_array))
            return max_xor

        result = []
        for l, r in queries:
            sub_array = nums[l:r+1]
            result.append(find_max_xor_score(sub_array))
        
        return result