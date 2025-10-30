class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def calculate_xor_score(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0]

            temp_arr = arr[:]
            while len(temp_arr) > 1:
                new_arr = []
                for i in range(len(temp_arr) - 1):
                    new_arr.append(temp_arr[i] ^ temp_arr[i+1])
                temp_arr = new_arr
            return temp_arr[0]

        def max_xor_subarray(arr):
            max_xor = 0
            for i in range(len(arr)):
                for j in range(i, len(arr)):
                    sub_array = arr[i:j+1]
                    max_xor = max(max_xor, calculate_xor_score(sub_array))
            return max_xor

        result = []
        for l, r in queries:
            sub_array = nums[l:r+1]
            result.append(max_xor_subarray(sub_array))
        return result