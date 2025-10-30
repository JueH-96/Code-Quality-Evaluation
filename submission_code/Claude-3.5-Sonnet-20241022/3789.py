class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def count_valid_subarrays(pairs):
            # Count subarrays that don't contain any conflicting pairs
            count = 0
            # Try all possible start positions
            for start in range(n):
                # Keep track of numbers in current subarray
                current = set()
                # Try all possible end positions
                for end in range(start, n):
                    # Add the current number to our set
                    current.add(end + 1)
                    
                    # Check if this subarray is valid
                    valid = True
                    for a, b in pairs:
                        if a in current and b in current:
                            valid = False
                            break
                    
                    if valid:
                        count += 1
                        
            return count
        
        max_subarrays = 0
        
        # Try removing each conflicting pair one at a time
        for i in range(len(conflictingPairs)):
            # Create new list without the current pair
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            # Count valid subarrays with this pair removed
            subarrays = count_valid_subarrays(remaining_pairs)
            max_subarrays = max(max_subarrays, subarrays)
            
        return max_subarrays