from math import gcd
from collections import defaultdict

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Count the frequency of each number in nums
        num_counts = defaultdict(int)
        for num in nums:
            num_counts[num] += 1
        
        # Create a list of unique numbers and sort them
        unique_nums = sorted(num_counts.keys())
        n = len(unique_nums)
        
        # Initialize a dictionary to store the frequency of each GCD
        gcd_counts = defaultdict(int)
        
        # Calculate the GCD for all pairs (i, j) where i <= j
        for i in range(n):
            for j in range(i, n):
                a = unique_nums[i]
                b = unique_nums[j]
                current_gcd = gcd(a, b)
                if i == j:
                    # If i == j, the number of pairs is C(count, 2)
                    count = num_counts[a]
                    if count >= 2:
                        gcd_counts[current_gcd] += count * (count - 1) // 2
                else:
                    # If i != j, the number of pairs is count_a * count_b
                    count_a = num_counts[a]
                    count_b = num_counts[b]
                    gcd_counts[current_gcd] += count_a * count_b
        
        # Sort the GCDs and prepare the cumulative counts
        sorted_gcds = sorted(gcd_counts.keys())
        cumulative_counts = []
        total = 0
        for g in sorted_gcds:
            total += gcd_counts[g]
            cumulative_counts.append(total)
        
        # For each query, find the corresponding GCD
        answer = []
        for q in queries:
            # Find the smallest index where cumulative_counts[index] > q
            low, high = 0, len(sorted_gcds) - 1
            result = 0
            while low <= high:
                mid = (low + high) // 2
                if cumulative_counts[mid] > q:
                    result = mid
                    high = mid - 1
                else:
                    low = mid + 1
            answer.append(sorted_gcds[result])
        
        return answer