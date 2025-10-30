from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def calculate_xor_score(subarray):
            """Calculate the XOR score of a subarray."""
            score = subarray[0]
            for i in range(1, len(subarray)):
                score = score ^ subarray[i]
            return score

        def calculate_subarray_xor_scores(subarray):
            """Calculate the XOR scores of all subarrays."""
            xor_scores = []
            for i in range(len(subarray)):
                for j in range(i, len(subarray)):
                    subarray_xor_score = calculate_xor_score(subarray[i:j+1])
                    xor_scores.append(subarray_xor_score)
            return xor_scores

        answer = []
        for query in queries:
            left, right = query
            subarray = nums[left:right+1]
            subarray_xor_scores = calculate_subarray_xor_scores(subarray)
            max_xor_score = max(subarray_xor_scores)
            answer.append(max_xor_score)
        return answer