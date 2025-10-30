class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        if not conflictingPairs:
            return n * (n + 1) // 2
        
        max_subarrays = 0
        
        # We need to try removing each conflicting pair once and compute the result for the remaining pairs
        for i in range(len(conflictingPairs)):
            # The remaining pairs are all except the i-th pair
            remaining_pairs = conflictingPairs[:i] + conflictingPairs[i+1:]
            
            # Now, for the remaining pairs, find all the forbidden pairs (a, b)
            # We need to find all intervals [l, r] where no a and b from any remaining pair are both present
            # Then, the valid subarrays are those entirely within such intervals
            
            # Collect all forbidden numbers and their positions
            forbidden = set()
            for a, b in remaining_pairs:
                forbidden.add(a)
                forbidden.add(b)
            
            # If there are no remaining pairs, all subarrays are valid
            if not remaining_pairs:
                total = n * (n + 1) // 2
                if total > max_subarrays:
                    max_subarrays = total
                continue
            
            # Create a list of positions for each forbidden number
            pos_map = {num: idx + 1 for idx, num in enumerate(range(1, n+1))}  # 1-based
            
            # Now, for each remaining pair (a, b), we need to note their positions
            # The idea is to split the array into segments where no segment contains both elements of any pair
            # The approach is similar to finding minimal segments that can be split
            
            # We can represent each pair as intervals between the min and max positions of the pair elements
            intervals = []
            for a, b in remaining_pairs:
                x, y = pos_map[a], pos_map[b]
                if x > y:
                    x, y = y, x
                intervals.append((x, y))
            
            # Now, merge overlapping intervals to find the minimal set of intervals that cover all conflicts
            if not intervals:
                total = n * (n + 1) // 2
                if total > max_subarrays:
                    max_subarrays = total
                continue
            
            # Sort intervals by start time
            intervals.sort()
            merged = []
            current_start, current_end = intervals[0]
            for s, e in intervals[1:]:
                if s <= current_end:
                    current_start = min(current_start, s)
                    current_end = max(current_end, e)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = s, e
            merged.append((current_start, current_end))
            
            # Now, the merged intervals represent regions where any subarray that spans them is invalid
            # Valid subarrays must lie completely between these intervals
            # The total valid subarrays is the sum of subarrays in the gaps between these intervals
            
            total_valid = 0
            prev_end = 0
            for start, end in merged:
                length = start - prev_end - 1
                if length > 0:
                    total_valid += length * (length + 1) // 2
                prev_end = end
            # Add the segment after the last interval
            length = n - prev_end
            if length > 0:
                total_valid += length * (length + 1) // 2
            
            if total_valid > max_subarrays:
                max_subarrays = total_valid
        
        return max_subarrays