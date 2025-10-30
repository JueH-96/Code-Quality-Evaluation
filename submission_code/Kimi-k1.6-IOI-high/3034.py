class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        # Sort the intervals by their start point
        nums.sort()
        merged = []
        for interval in nums:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = interval
                # Check if the current interval overlaps or is adjacent to the last merged interval
                if current_start <= last_end:
                    # Merge the intervals
                    new_start = last_start
                    new_end = max(last_end, current_end)
                    merged[-1] = [new_start, new_end]
                else:
                    merged.append(interval)
        # Calculate the total number of points covered
        total = 0
        for start, end in merged:
            total += end - start + 1
        return total