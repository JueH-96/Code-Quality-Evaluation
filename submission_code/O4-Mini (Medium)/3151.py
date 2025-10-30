from typing import List

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # We need to assign 4n tasks to n processors (each has 4 cores),
        # so that each processor i with ready time r_i gets exactly 4 tasks,
        # and its finish time is r_i + max(its tasks). We minimize the
        # overall makespan (maximum finish time).
        #
        # We binary‐search the minimal T such that we can assign all tasks
        # under the constraint that for each processor i, all its tasks t
        # satisfy r_i + t <= T  <=>  t <= T - r_i.  Each processor has capacity 4.
        #
        # For a candidate T, define d_i = T - r_i (the maximum task size processor i can take).
        # We collect all 4-slot capacities d_i (each repeated 4 times),
        # sort them ascending, and try to match tasks sorted ascending greedily:
        # assign each smallest task to the smallest available slot >= task size.
        # If all tasks can be matched, T is feasible.
        
        n = len(processorTime)
        m = len(tasks)
        # Sort tasks ascending once
        tasks_sorted = sorted(tasks)
        # Sort processor times ascending, then we'll derive d_i in ascending order per check
        r_sorted = sorted(processorTime)
        
        # Binary search for minimal T
        lo = 0
        hi = max(r_sorted) + tasks_sorted[-1]
        
        # Feasibility check for a given T
        def feasible(T: int) -> bool:
            # Build d_i = T - r_i, and we want them in ascending order
            # r_sorted is ascending, so T - r_sorted is descending.
            # So reverse r_sorted to get descending r, hence ascending d.
            d_desc = [T - r for r in reversed(r_sorted)]
            # We'll greedily match tasks_sorted (ascending) to these slots.
            pi = 0
            rem = [4] * n  # each processor has 4 slots
            for t in tasks_sorted:
                # advance to a processor whose slot can take t
                while pi < n and (d_desc[pi] < t or rem[pi] == 0):
                    pi += 1
                if pi == n:
                    return False
                rem[pi] -= 1
            return True
        
        # Standard binary search on integer range
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                hi = mid
            else:
                lo = mid + 1
        
        return lo

# Example usage:
# sol = Solution()
# print(sol.minProcessingTime([8,10], [2,2,3,1,8,7,4,5]))  # Expected 16
# print(sol.minProcessingTime([10,20], [2,3,1,2,5,8,4,3]))  # Expected 23