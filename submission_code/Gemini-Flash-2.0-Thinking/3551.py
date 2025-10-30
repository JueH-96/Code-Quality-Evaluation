class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def calculate_xor_score(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0]
            current_arr = list(arr)
            while len(current_arr) > 1:
                next_arr = []
                for k in range(len(current_arr) - 1):
                    next_arr.append(current_arr[k] ^ current_arr[k + 1])
                current_arr = next_arr
            return current_arr[0]

        results = []
        for l, r in queries:
            max_xor = 0
            sub_array = nums[l : r + 1]
            n = len(sub_array)
            for i in range(n):
                for j in range(i, n):
                    current_subarray = sub_array[i : j + 1]
                    xor_score = calculate_xor_score(current_subarray)
                    max_xor = max(max_xor, xor_score)
            results.append(max_xor)
        return results