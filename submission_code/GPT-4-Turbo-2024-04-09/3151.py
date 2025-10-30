class Solution:
    def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
        import heapq
        
        # Number of processors
        n = len(processorTime)
        
        # Sort processors by their available time
        heapq.heapify(processorTime)
        
        # Sort tasks in descending order to prioritize longer tasks
        tasks.sort(reverse=True)
        
        # Create a list to store the maximum time each processor will finish
        processor_finish_times = [0] * n
        
        # Assign tasks to processors
        for i in range(n):
            # Get the earliest available processor
            earliest_start_time = heapq.heappop(processorTime)
            
            # Calculate finish times for the next 4 tasks assigned to this processor
            max_finish_time = 0
            for j in range(4):
                task_index = 4 * i + j
                finish_time = earliest_start_time + tasks[task_index]
                max_finish_time = max(max_finish_time, finish_time)
            
            # Store the maximum finish time for this processor
            processor_finish_times[i] = max_finish_time
            
            # Push the finish time back into the heap
            heapq.heappush(processorTime, max_finish_time)
        
        # The result is the maximum of all processor finish times
        return max(processor_finish_times)