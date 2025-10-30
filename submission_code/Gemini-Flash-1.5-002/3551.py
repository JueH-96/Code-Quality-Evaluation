class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def xor_score(arr):
            if not arr:
                return 0
            while len(arr) > 1:
                new_arr = []
                for i in range(len(arr) - 1):
                    new_arr.append(arr[i] ^ arr[i+1])
                arr = new_arr
            return arr[0]

        ans = []
        for l, r in queries:
            max_xor = 0
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    max_xor = max(max_xor, xor_score(nums[i:j+1]))
            ans.append(max_xor)
        return ans