class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        q = len(queries)
        ans = [0] * q
        for i in range(q):
            l, r = queries[i]
            max_xor = 0
            for j in range(l, r + 1):
                for k in range(j, r + 1):
                    sub_array = nums[j:k+1]
                    xor_score = 0
                    while len(sub_array) > 1:
                        new_sub_array = []
                        for m in range(len(sub_array) - 1):
                            new_sub_array.append(sub_array[m] ^ sub_array[m+1])
                        sub_array = new_sub_array
                    if len(sub_array) == 1:
                        xor_score = sub_array[0]
                    max_xor = max(max_xor, xor_score)
            ans[i] = max_xor
        return ans