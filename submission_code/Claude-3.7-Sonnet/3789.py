class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_valid_subarrays(pairs):
            total = L = 0
            # For each subarray of nums
            for i in range(n):
                for j in range(i, n):
                    valid = True
                    # Check if subarray contains both elements of any conflicting pair
                    for a, b in pairs:
                        if i+1 <= a <= j+1 and i+1 <= b <= j+1:
                            valid = False
                            break
                    if valid:
                        total += 1
            return total
        
        max_subarrays = 0
        # Try removing each conflicting pair
        for i in range(len(conflictingPairs)):
            # Create new list without the i-th pair
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            count = count_valid_subarrays(remaining_pairs)
            max_subarrays = max(max_subarrays, count)
        
        return max_subarrays