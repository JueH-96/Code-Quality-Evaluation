import math
from collections import defaultdict

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Count the frequency of each number in nums
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Create a list of unique numbers and sort them
        unique_nums = sorted(freq.keys())
        n = len(unique_nums)
        
        # Initialize a dictionary to count the frequency of each GCD
        gcd_count = defaultdict(int)
        
        # Iterate over all pairs (i, j) where i <= j
        for i in range(n):
            for j in range(i, n):
                a = unique_nums[i]
                b = unique_nums[j]
                current_gcd = math.gcd(a, b)
                if i == j:
                    # If i == j, the number of pairs is C(freq[a], 2)
                    count = freq[a] * (freq[a] - 1) // 2
                else:
                    # If i != j, the number of pairs is freq[a] * freq[b]
                    count = freq[a] * freq[b]
                gcd_count[current_gcd] += count
        
        # Convert the gcd_count dictionary to a sorted list of (gcd, count)
        sorted_gcds = sorted(gcd_count.items())
        
        # Create a list of all GCDs in sorted order, considering their counts
        all_gcds = []
        for gcd_val, count in sorted_gcds:
            all_gcds.extend([gcd_val] * count)
        
        # Sort the all_gcds list to get the final gcdPairs
        all_gcds.sort()
        
        # Prepare the answer for each query
        answer = []
        for q in queries:
            answer.append(all_gcds[q])
        
        return answer