from typing import List
from collections import defaultdict


class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        n = len(nums)
        
        prefixItemCntLess = [0]*n
        prefixItemCntLessEq = [0]*n
        postfixItemCntLess = [0]*n
        postfixItemCntLessEq = [n]*n
        
        numsSortedWithIdx = sorted([(nums[i], i) for i in range(n)])
        for i in range(n):
            num, idx = numsSortedWithIdx[i]
            if i > 0:
                prefixItemCntLess[idx] = prefixItemCntLess[numsSortedWithIdx[i-1][1]] + (numsSortedWithIdx[i-1][0] < num)
                prefixItemCntLessEq[idx] = prefixItemCntLessEq[numsSortedWithIdx[i-1][1]] + (numsSortedWithIdx[i-1][0] <= num)
                
        for i in range(n-1, -1, -1):
            num, idx = numsSortedWithIdx[i]
            if i < n-1:
                postfixItemCntLess[idx] = postfixItemCntLess[numsSortedWithIdx[i+1][1]] + (numsSortedWithIdx[i+1][0] < num)
                postfixItemCntLessEq[idx] = postfixItemCntLessEq[numsSortedWithIdx[i+1][1]] - (numsSortedWithIdx[i+1][0] >= num)
        
        s = 0
        p = 10**9 + 7
        for i in range(n):
            num = nums[i]
            s += num**3 * (postfixItemCntLessEq[i] - prefixItemCntLessEq[i])
            s += num**2 * num * ((n - 1 - postfixItemCntLessEq[i]) + prefixItemCntLess[i])
            s %= p

        return s