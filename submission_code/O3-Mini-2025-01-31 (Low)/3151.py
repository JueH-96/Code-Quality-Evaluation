from typing import List
import bisect

class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        # We have n processors with 4 cores each and 4*n tasks.
        # For a given candidate "T" (global finishing time), a processor with available time p 
        # can only complete a task of duration t if p + t <= T, i.e. t <= T - p.
        # Since each processor runs its 4 tasks concurrently (one per core), its finish time is p + max(task assigned).
        # Hence, a processor i can only be assigned tasks of duration <= (T - processorTime[i]).
        # We need to decide if it is possible to assign all tasks to the processors, 
        # where each processor gets 4 tasks, and each task t goes to a processor 
        # with capacity >= t.  Note that a processor’s “capacity” is T - p.
        #
        # Observing that the only effect of the assignment is that for each processor 
        # the maximum t among its 4 tasks is <= (T - p). Importantly, if a processor gets 
        # one “hard” task (nearly T - p), then the other three tasks can be arbitrarily low 
        # (and can even be “hard” for some other processor) as long as that processor can handle them.
        #
        # Hence, a necessary and sufficient condition is that all tasks can be “packed”
        # into the available slots of the processors, where processor i supplies 4 slots and 
        # each slot can only accept a task of duration <= (T - processorTime[i]).
        #
        # To check this condition efficiently we do the following:
        #  1. For each processor i, define its capacity = T - processorTime[i].
        #  2. Only those tasks with t <= capacity can be assigned to that processor.
        #  3. We try to assign tasks in descending order (largest tasks first) to ensure that
        #     the most “demanding” tasks get placed into some slot that can handle them.
        #
        # We maintain a sorted list of available processor slots keyed by capacity.
        # Each processor contributes 4 slots (all having the same capacity).
        # Then for each task (processing from largest to smallest) we find the first available slot 
        # with capacity >= task (using binary search). If found, we use that slot (decreasing available count for that processor).
        # If not found, the candidate T fails.
        #
        # We then binary search over T. Lower bound is at least: 
        #    min_possible_T = min(processorTime) + min(tasks)   (if processors with low start time handle the smallest tasks)
        # and upper bound is:
        #    max_possible_T = max(processorTime) + max(tasks)   (if the slowest processor gets the hardest task).
        
        n = len(processorTime)
        tasksCount = len(tasks)
        # Sanity: tasksCount should equal 4*n.
        if tasksCount != 4 * n:
            raise ValueError("tasks length must be 4 times the number of processors")
        
        min_proc = min(processorTime)
        max_proc = max(processorTime)
        min_task = min(tasks)
        max_task = max(tasks)
        
        low = min_proc + min_task
        high = max_proc + max_task
        
        # Sort tasks descending once (we iterate on tasks in descending order in our feasibility check).
        tasks_sorted = sorted(tasks, reverse=True)
        
        # We'll prepare arrays for processor available times inside the check function.
        def canFinish(T: int) -> bool:
            # For each processor, capacity = T - p.
            # A processor cannot handle any task with duration > (T - p).
            # We need to assign each task from tasks_sorted to a "slot" from some processor.
            # We set up a sorted list (by capacity) of entries: each entry is [capacity, available_slots].
            # Initially, for each processor with processing time p, if T - p >= 0, add a slot with capacity cap = T - p and available count = 4.
            # (If T - p < 0 then processor cannot take any task and T is infeasible.)
            processors = []
            for p in processorTime:
                cap = T - p
                if cap < 0:
                    return False  # This processor cannot even start a minimal task.
                # We'll store as a tuple (cap, available_slots). They will be sorted by cap.
                processors.append([cap, 4])
            processors.sort(key=lambda x: x[0])
            
            # We'll maintain a list of capacities (the first element of each entry) for binary search.
            caps = [entry[0] for entry in processors]
            
            # Now, iterate over tasks in descending order.
            for t in tasks_sorted:
                # Use binary search on 'caps' to find the leftmost processor with cap >= t.
                # Note: even if the processor at that index has 0 available slots, then in our structure, we remove it.
                idx = bisect.bisect_left(caps, t)
                if idx == len(caps):
                    return False  # no available processor slot can handle this task.
                # Use this slot: decrement available count.
                processors[idx][1] -= 1
                if processors[idx][1] == 0:
                    # Remove this processor slot from our structure.
                    del processors[idx]
                    del caps[idx]
            return True
        
        # Binary search for the minimum T that returns True.
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if canFinish(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans