class Solution:
    def minOperations(self, nums: List[int]) -> int:
        import sys
        import math

        MAX = 10**6 + 1
        SPF = [0] * MAX  # Smallest Prime Factor
        SPF[0] = SPF[1] = 0  # 0 and 1 have no prime factors
        for i in range(2, MAX):
            if SPF[i] == 0:
                SPF[i] = i
                for j in range(i * 2, MAX, i):
                    if SPF[j] == 0:
                        SPF[j] = i

        operations = 0
        required_upper = float('inf')
        for x in reversed(nums):
            if x <= required_upper:
                # Keep x as is
                assigned = x
                # No operation needed
            else:
                if x >=2 and SPF[x] != x:
                    # Can replace with smallest prime factor
                    p = SPF[x]
                    if p <= required_upper:
                        assigned = p
                        operations +=1
                    else:
                        return -1
                else:
                    # Cannot replace (x is prime or x=1)
                    return -1
            required_upper = assigned
        return operations