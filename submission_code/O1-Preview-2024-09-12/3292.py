class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        from collections import defaultdict
        import heapq

        # Convert changeIndices to 1-based index
        # Map each index i to the list of times s when changeIndices[s] == i
        index_times = [[] for _ in range(n + 1)]  # indices from 1 to n

        for s, idx in enumerate(changeIndices, 1):
            # idx in [1, n]
            index_times[idx].append(s)

        # For each index i, we need to find earliest mark time
        earliest_mark_time = [0] * (n + 1)
        latest_start_time = [0] * (n + 1)
        nums = [0] + nums  # make nums 1-indexed
        mark_scheduled = [False] * (n + 1)
        total_decrements = sum(nums[1:])

        for i in range(1, n + 1):
            if not index_times[i]:
                # Cannot mark index i
                return -1
            D_i = nums[i]
            # Earliest time we can complete D_i decrements is D_i
            possible_mark_times = [s for s in index_times[i] if s >= D_i]
            if not possible_mark_times:
                return -1
            earliest_mark_time[i] = min(possible_mark_times)
            latest_start_time[i] = earliest_mark_time[i] - D_i + 1  # Start from this time
            if latest_start_time[i] < 1:
                return -1

        # Total operations required: total_decrements + n mark operations
        total_ops_required = total_decrements + n
        if total_ops_required > m:
            return -1

        # Now, proceed to schedule tasks
        # Build events for each time s
        events = defaultdict(list)  # events at each time s
        for i in range(1, n + 1):
            # At time latest_start_time[i], we start decrement tasks for index i
            events[latest_start_time[i]].append(('start_decrement', i))
            # At time earliest_mark_time[i], we may schedule mark task
            events[earliest_mark_time[i]].append(('possible_mark', i))
            # For mark times beyond earliest_mark_time[i], we add possible marks
            for s in index_times[i]:
                if s > earliest_mark_time[i]:
                    events[s].append(('possible_mark', i))

        # Need to schedule decrement tasks within [latest_start_time[i], earliest_mark_time[i]-1]
        # For each index i, remaining_decrements[i] counts remaining decrements
        remaining_decrements = {}
        decrement_intervals = {}
        for i in range(1, n + 1):
            remaining_decrements[i] = nums[i]
            decrement_intervals[i] = (latest_start_time[i], earliest_mark_time[i] - 1)

        # Initialize min-heap for decrement tasks (order by earliest_mark_time)
        import heapq
        decrement_heap = []
        s = 1
        max_time_needed = 0
        while s <= m:
            # Process events at time s
            if s in events:
                for event in events[s]:
                    if event[0] == 'start_decrement':
                        i = event[1]
                        # Add index i to heap
                        heapq.heappush(decrement_heap, (earliest_mark_time[i], i))
                    elif event[0] == 'possible_mark':
                        i = event[1]
                        if not mark_scheduled[i] and nums[i] == 0:
                            # Can schedule mark task for index i
                            mark_scheduled[i] = True
                            max_time_needed = s
                            break  # Only one operation per time
            if not (s in events and any(event[0]=='possible_mark' and not mark_scheduled[event[1]] and nums[event[1]]==0 for event in events[s])):
                # No mark task scheduled, try to schedule decrement task
                while decrement_heap:
                    _, i = heapq.heappop(decrement_heap)
                    l, r = decrement_intervals[i]
                    if l <= s <= r and remaining_decrements[i] > 0:
                        # Schedule decrement task for index i
                        nums[i] -=1
                        remaining_decrements[i] -=1
                        if remaining_decrements[i] > 0:
                            # Push back into heap for further decrements
                            heapq.heappush(decrement_heap, (earliest_mark_time[i], i))
                        break  # Only one operation per time
                else:
                    # No decrement tasks to schedule
                    pass  # Do nothing
            s +=1

        # After scheduling, check if all indices are marked
        if all(mark_scheduled[1:]):
            return max_time_needed
        else:
            return -1