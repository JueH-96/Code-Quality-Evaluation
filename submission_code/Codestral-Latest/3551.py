class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def xor_score(arr):
            while len(arr) > 1:
                new_arr = []
                for i in range(len(arr) - 1):
                    new_arr.append(arr[i] ^ arr[i + 1])
                arr = new_arr
            return arr[0]

        def max_xor_score_in_range(l, r):
            max_score = 0
            for i in range(l, r + 1):
                for j in range(i, r + 1):
                    subarray = nums[i:j + 1]
                    score = xor_score(subarray)
                    max_score = max(max_score, score)
            return max_score

        answer = []
        for l, r in queries:
            answer.append(max_xor_score_in_range(l, r))

        return answer