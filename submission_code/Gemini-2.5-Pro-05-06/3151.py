from typing import List

class Solution:
  def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
    # Sort processorTime in ascending order.
    # Processors that become available earlier are at the beginning of the list.
    processorTime.sort()
    
    # Sort tasks in ascending order.
    # Shorter tasks are at the beginning, longer tasks at the end.
    tasks.sort()
    
    n = len(processorTime)
    # tasks.length is guaranteed to be 4 * n
    
    max_overall_completion_time = 0
    
    # Iterate through processors, from earliest available (processorTime[0])
    # to latest available (processorTime[n-1]).
    
    # The greedy strategy pairs the earliest available processor with the group
    # of 4 tasks that includes the 4 tasks with longest durations overall.
    # The next earliest processor gets the next 4 longest tasks, and so on.
    
    task_list_len = len(tasks) # This is 4*n
    
    for i in range(n):
      current_processor_availability_time = processorTime[i]
      
      # The i-th processor (0-indexed in sorted processorTime list) is processorTime[i].
      # It is assigned 4 tasks. The tasks are chosen from the higher end of the sorted 'tasks' list.
      # The group of 4 tasks for processorTime[i] are (in terms of their original indices in sorted tasks list):
      #   tasks[task_list_len - 1 - (4*i)]  (This is the longest task of the four assigned to processorTime[i])
      #   tasks[task_list_len - 2 - (4*i)]
      #   tasks[task_list_len - 3 - (4*i)]
      #   tasks[task_list_len - 4 - (4*i)]  (This is the shortest task of the four assigned to processorTime[i])
      #
      # The completion time for this processor is determined by its availability time plus the duration of 
      # the longest task it executes.
      max_task_duration_for_this_processor = tasks[task_list_len - 1 - (4 * i)]
      
      completion_time_for_this_processor = current_processor_availability_time + max_task_duration_for_this_processor
      
      # We want to find the minimum time by which ALL tasks are completed.
      # This means we need to find the maximum completion time among all processors,
      # and our greedy assignment aims to minimize this value.
      if completion_time_for_this_processor > max_overall_completion_time:
        max_overall_completion_time = completion_time_for_this_processor
        
    return max_overall_completion_time