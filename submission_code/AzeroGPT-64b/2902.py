from typing import List
class Solution:
    def maxSum(self, nums: List[int]) -> int:
        maxDigits = [float('-inf')] * 10
        for num in nums:
            digit = max(int(d) for d in str(num))
            maxDigits[digit] = max(maxDigits[digit], num)
        
        result = float('-inf')
        for i in range(10):
            if maxDigits[i] != float('-inf') and maxDigits[i] != None:
                for j in range(i, 10):
                    if j != i and maxDigits[j] == float('-inf'):
                        continue
                    result = max(result, maxDigits[i] + maxDigits[j])
        return result if result != float('-inf') else -1