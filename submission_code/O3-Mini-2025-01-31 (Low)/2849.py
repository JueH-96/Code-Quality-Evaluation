from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        
        # iterate over all subarrays starting at index i
        for i in range(n):
            # We will maintain the intervals of consecutive numbers that appear in nums[i..j].
            # We use two dictionaries:
            #   seg_by_start: {start: end} for an interval [start, end].
            #   seg_by_end: {end: start} for the same intervals.
            seg_by_start = {}
            seg_by_end = {}
            seg_count = 0   # current number of intervals
            
            for j in range(i, n):
                x = nums[j]
                # We add number x if not already present.
                # (Since a duplicate does not change our set of distinct numbers, we only do work when x is new)
                # If it is already present then the subarray's distinct set (and hence its intervals) remains the same.
                # We check if x is already in one of our intervals.
                # A number x is in the union of the intervals if there is an interval [s, e] such that s <= x <= e.
                already_present = False
                # check if x is exactly the start of an interval or falls in an interval
                # Since intervals are non‐overlapping and sorted by their keys in seg_by_start we can simply check:
                #   if there exists a key s such that s <= x and the corresponding e >= x.
                # For simplicity, we simply check membership by checking the neighbors x-1 and x+1:
                # Actually, note that if x was already added it must lie in one of the intervals.
                if x in seg_by_start or x in seg_by_end:
                    already_present = True
                else:
                    # also we can check one of the intervals if x falls in them.
                    # But note, our intervals are “maximal” and if x falls inside an interval, then either x==s or x==e or
                    # s < x < e. We can check x-1 in seg_by_end: if x-1 exists then x is already consecutive?
                    # However, a simpler trick is to store an extra boolean table for seen numbers,
                    # since the maximum value is n (i.e. up to 1000) so we can use a simple array.
                    pass  # we handle duplicates below
                
                # Here, we choose an alternative: we maintain a seen array for the subarray.
                # (Because duplicates do not alter the set of distinct numbers.)
                # We can simply simulate that by checking if x has been "added" before.
                # We'll create an array of length (n+2) for convenience.
                # To avoid re-allocation every time in the inner loop we simulate it using seg_by_start and seg_by_end.
                # However, a simple check is to see if there is an interval that already covers x.
                # We do that with a manual check: since the intervals cover disjoint ranges we try:
                duplicated = False
                # Check if any interval already covers x:
                # The easiest is to check if x-1 is in seg_by_end
                if not seg_by_start and not seg_by_end:
                    # if there is no interval, then certainly not duplicate.
                    duplicated = False
                else:
                    # Because n is small, we can iterate over the intervals.
                    for s, e in seg_by_start.items():
                        if s <= x <= e:
                            duplicated = True
                            break
                
                if duplicated:
                    # x is duplicate, so no change in intervals.
                    total_imbalance += (seg_count - 1) if seg_count > 0 else 0
                    continue
                
                # Now x is a new number in the subarray.
                # Check if there is an interval ending at x-1.
                left_interval_start = seg_by_end.get(x - 1, None)
                # Check if there is an interval starting at x+1.
                right_interval_end = seg_by_start.get(x + 1, None)
                
                if left_interval_start is None and right_interval_end is None:
                    # x stands alone. Create a new interval [x, x].
                    seg_by_start[x] = x
                    seg_by_end[x] = x
                    seg_count += 1
                elif left_interval_start is not None and right_interval_end is None:
                    # Extend the left interval which ends with x-1.
                    # Remove the old mapping for the end.
                    del seg_by_end[x - 1]
                    new_end = x
                    seg_by_start[left_interval_start] = new_end
                    seg_by_end[new_end] = left_interval_start
                elif left_interval_start is None and right_interval_end is not None:
                    # Extend the right interval which starts with x+1.
                    # Remove the old mapping for the start.
                    del seg_by_start[x + 1]
                    new_start = x
                    seg_by_start[new_start] = right_interval_end
                    seg_by_end[right_interval_end] = new_start
                else:
                    # Both exist: we can merge the left interval and the right interval with x bridging them.
                    # left interval is [left_interval_start, x-1]
                    # right interval is [x+1, right_interval_end]
                    # Merge them into [left_interval_start, right_interval_end].
                    del seg_by_end[x - 1]
                    del seg_by_start[x + 1]
                    seg_by_start[left_interval_start] = right_interval_end
                    seg_by_end[right_interval_end] = left_interval_start
                    # We merged two intervals into one so decrease count.
                    seg_count -= 1
                
                # Now imbalance for the current subarray is just number of intervals minus one.
                current_imbalance = (seg_count - 1) if seg_count > 0 else 0
                total_imbalance += current_imbalance
        
        return total_imbalance