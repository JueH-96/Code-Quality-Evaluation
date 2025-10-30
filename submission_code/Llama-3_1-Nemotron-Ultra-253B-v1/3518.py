from typing import List
import sys

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        if n < 4:
            return 0  # Not possible as per constraints, but handle edge case
        
        # Precompute step3 contribution
        max_step3 = [-sys.maxsize] * n
        min_step3 = [sys.maxsize] * n
        for j in range(n-2, -1, -1):
            max_step3[j] = max(b[j+1], max_step3[j+1])
            min_step3[j] = min(b[j+1], min_step3[j+1])
        # Handle j = n-1 case (out of bounds)
        max_step3[-1] = -sys.maxsize
        min_step3[-1] = sys.maxsize
        
        step3_contribution = []
        for j in range(n):
            if a[3] >= 0:
                val = max_step3[j]
            else:
                val = min_step3[j]
            step3_contribution.append(a[3] * val)
        
        # Precompute step2 values and suffix arrays
        step2_values = []
        for i2 in range(n):
            if i2 <= n-2:
                val = a[2] * b[i2] + step3_contribution[i2]
            else:
                val = -sys.maxsize if a[2] >= 0 else sys.maxsize
            step2_values.append(val)
        
        suffix_max_step2 = [-sys.maxsize] * n
        suffix_min_step2 = [sys.maxsize] * n
        if n >= 2:
            suffix_max_step2[n-2] = step2_values[n-2]
            suffix_min_step2[n-2] = step2_values[n-2]
            for i in range(n-3, -1, -1):
                suffix_max_step2[i] = max(step2_values[i], suffix_max_step2[i+1])
                suffix_min_step2[i] = min(step2_values[i], suffix_min_step2[i+1])
        
        # Precompute step2 contribution
        step2_contribution = []
        for j in range(n):
            if j + 1 > n - 2:
                val = -sys.maxsize if a[2] >= 0 else sys.maxsize
            else:
                if a[2] >= 0:
                    val = suffix_max_step2[j + 1]
                else:
                    val = suffix_min_step2[j + 1]
            step2_contribution.append(val)
        
        # Precompute step1 values and suffix arrays
        step1_values = []
        for i1 in range(n):
            if i1 <= n - 3:
                val = a[1] * b[i1] + step2_contribution[i1]
            else:
                val = -sys.maxsize if a[1] >= 0 else sys.maxsize
            step1_values.append(val)
        
        suffix_max_step1 = [-sys.maxsize] * n
        suffix_min_step1 = [sys.maxsize] * n
        if n >= 3:
            suffix_max_step1[n-3] = step1_values[n-3]
            suffix_min_step1[n-3] = step1_values[n-3]
            for i in range(n-4, -1, -1):
                suffix_max_step1[i] = max(step1_values[i], suffix_max_step1[i+1])
                suffix_min_step1[i] = min(step1_values[i], suffix_min_step1[i+1])
        
        # Precompute step1 contribution
        step1_contribution = []
        for j in range(n):
            if j + 1 > n - 3:
                val = -sys.maxsize if a[1] >= 0 else sys.maxsize
            else:
                if a[1] >= 0:
                    val = suffix_max_step1[j + 1]
                else:
                    val = suffix_min_step1[j + 1]
            step1_contribution.append(val)
        
        # Precompute step0 values
        step0_values = []
        for i0 in range(n):
            if i0 <= n - 4:
                val = a[0] * b[i0] + step1_contribution[i0]
            else:
                val = -sys.maxsize if a[0] >= 0 else sys.maxsize
            step0_values.append(val)
        
        # Find the maximum value in step0_values[0...n-4]
        max_score = max(step0_values[:n-3])
        
        return max_score