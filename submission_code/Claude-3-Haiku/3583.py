from math import gcd
from collections import defaultdict

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        gcd_pairs = []
        gcd_dict = defaultdict(list)

        # Calculate GCD of all possible pairs and store them in a sorted list
        for i in range(n):
            for j in range(i+1, n):
                g = gcd(nums[i], nums[j])
                gcd_pairs.append(g)
                gcd_dict[g].append(len(gcd_pairs) - 1)
        gcd_pairs.sort()

        # Retrieve the values at the given indices in the sorted gcd_pairs list
        answer = []
        for q in queries:
            if q < len(gcd_pairs):
                answer.append(gcd_pairs[q])
            else:
                # Find the index of the qth element in the sorted gcd_pairs list
                idx = next((i for i in gcd_dict[gcd_pairs[q]] if i >= q), None)
                if idx is not None:
                    answer.append(gcd_pairs[idx])
                else:
                    answer.append(-1)
        return answer