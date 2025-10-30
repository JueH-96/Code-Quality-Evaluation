import math
from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        positives = [num for num in nums if num > 0]
        negatives = [num for num in nums if num < 0]
        pos_count = len(positives)
        neg_count = len(negatives)
        
        if pos_count > 0:
            pos_prod = 1
            for num in positives:
                pos_prod *= num
            neg_prod = 1
            for num in negatives:
                neg_prod *= num
            if neg_count % 2 == 0:
                result = pos_prod * neg_prod
            else:
                if neg_count == 0:
                    result = pos_prod
                else:
                    min_abs_neg = min(negatives, key=lambda x: abs(x))
                    result = pos_prod * (neg_prod // min_abs_neg)
            return result
        else:  # pos_count == 0
            if neg_count >= 2:
                neg_prod = 1
                for num in negatives:
                    neg_prod *= num
                if neg_count % 2 == 0:
                    return neg_prod
                else:
                    min_abs_neg = min(negatives, key=lambda x: abs(x))
                    return neg_prod // min_abs_neg
            else:  # neg_count < 2
                return max(nums)