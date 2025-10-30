class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        def countValidSubarrays(pairs):
            valid = 0
            
            # For each starting position
            for start in range(1, n + 1):
                # For each ending position
                for end in range(start, n + 1):
                    # Check if this subarray is valid
                    is_valid = True
                    for a, b in pairs:
                        if start <= a <= end and start <= b <= end:
                            is_valid = False
                            break
                    
                    if is_valid:
                        valid += 1
            
            return valid
        
        max_count = 0
        
        # Try removing each conflicting pair
        for i in range(len(conflictingPairs)):
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            count = countValidSubarrays(remaining_pairs)
            max_count = max(max_count, count)
        
        return max_count