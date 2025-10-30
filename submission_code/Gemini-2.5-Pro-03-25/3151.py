import math
from typing import List

class Solution:
  """
  Represents a solution to find the minimum processing time for tasks on processors.
  """
  def minProcessingTime(self, processorTime: List[int], tasks: List[int]) -> int:
    """
    Calculates the minimum time required to finish executing all tasks across all processors.

    Problem Setup:
    - There are 'n' processors, each with 4 cores.
    - There are '4*n' tasks to be executed.
    - Each core executes exactly one task.
    - `processorTime[i]` is the time the i-th processor becomes available.
    - `tasks[j]` is the duration of the j-th task.
    - Tasks assigned to a processor start only when the processor is available.
    - The 4 cores of a processor work independently and in parallel.
    - We need to find an assignment of tasks to processors that minimizes the time the last task finishes.

    Optimal Strategy:
    To minimize the overall completion time (the time the last task finishes), we should try to balance the workload and avoid situations where a processor that becomes available late is assigned a very long task. The optimal strategy is a greedy approach:
    1. Sort the processors based on their available times in ascending order. Processors that are available earlier should potentially handle longer tasks.
    2. Sort the tasks based on their durations in descending order. The longest tasks should be assigned first.
    3. Assign the 4 longest tasks to the processor that becomes available earliest. Assign the next 4 longest tasks to the second earliest available processor, and so on.
    
    Why this works:
    Consider any two processors `i` and `j` such that processor `i` becomes available earlier than or at the same time as processor `j` ($P_i \le P_j$). If processor `i` is assigned a task $t_i$ and processor `j` is assigned a task $t_j$ such that $D(t_i) < D(t_j)$, swapping these tasks can potentially reduce the maximum completion time or keep it the same. Specifically, assigning longer tasks to earlier processors allows them more time cushion, potentially reducing the overall maximum finish time. This logic leads to the strategy where the earliest processor gets the longest tasks, the second earliest gets the next longest, etc.

    Calculation Details:
    - Let the sorted available times be $P_0 \le P_1 \le \dots \le P_{n-1}$.
    - Let the sorted task durations be $D_0 \ge D_1 \ge \dots \ge D_{4n-1}$.
    - Processor `i` (the one available at time $P_i$) is assigned tasks with durations $D_{4i}, D_{4i+1}, D_{4i+2}, D_{4i+3}$.
    - Since the processor has 4 cores, these tasks run in parallel starting at time $P_i$.
    - The tasks finish at times $P_i + D_{4i}$, $P_i + D_{4i+1}$, $P_i + D_{4i+2}$, $P_i + D_{4i+3}$.
    - The time processor `i` finishes all its assigned tasks is $\max(P_i + D_{4i}, P_i + D_{4i+1}, P_i + D_{4i+2}, P_i + D_{4i+3})$.
    - This simplifies to $P_i + \max(D_{4i}, D_{4i+1}, D_{4i+2}, D_{4i+3})$.
    - Since the tasks are sorted in descending order, $D_{4i}$ is the maximum duration among the four tasks assigned to processor `i`.
    - So, the completion time for processor `i` is $C_i = P_i + D_{4i}$.
    - The overall minimum completion time is the maximum of these completion times over all processors: $\max_{i=0}^{n-1} C_i = \max_{i=0}^{n-1} (P_i + D_{4i})$.

    Args:
        processorTime: A list of integers where processorTime[i] is the time the i-th processor 
                       becomes available. There are 'n' processors.
        tasks: A list of integers representing the time it takes to execute each task. 
               There are '4*n' tasks.

    Returns:
        The minimum time by which all tasks have been executed.
    """
    
    # Get the number of processors, n.
    n = len(processorTime)
    
    # Sort processor available times in ascending order.
    # This ensures processorTime[i] corresponds to the (i+1)-th earliest available processor.
    processorTime.sort()
    
    # Sort task durations in descending order.
    # This ensures tasks[j] corresponds to the (j+1)-th longest task.
    tasks.sort(reverse=True)
    
    # Initialize the overall maximum completion time found so far.
    # This variable will store the time when the last task finishes across all processors.
    max_completion_time = 0
    
    # Iterate through each processor, indexed from 0 to n-1 based on their sorted availability.
    for i in range(n):
        # Processor `i` (in the sorted list) becomes available at time `processorTime[i]`.
        
        # According to the optimal strategy, processor `i` is assigned the block of 4 tasks
        # starting at index 4*i in the descending sorted `tasks` list.
        # The tasks are: tasks[4*i], tasks[4*i+1], tasks[4*i+2], tasks[4*i+3].
        
        # The duration of the longest task assigned to processor `i` is tasks[4*i] because the list is sorted descendingly.
        # The completion time for processor `i` is its available time plus the duration of its longest task.
        current_processor_completion_time = processorTime[i] + tasks[4*i]
        
        # The overall minimum time required to finish all tasks is determined by the processor that finishes last.
        # We update `max_completion_time` to track the maximum completion time encountered so far.
        max_completion_time = max(max_completion_time, current_processor_completion_time)
            
    # After iterating through all processors, `max_completion_time` holds the result.
    return max_completion_time