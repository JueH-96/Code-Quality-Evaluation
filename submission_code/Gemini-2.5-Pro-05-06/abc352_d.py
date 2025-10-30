import sys
from collections import deque

def main():
    N, K = map(int, sys.stdin.readline().split())
    P_list = list(map(int, sys.stdin.readline().split()))

    if K == 1:
        # For K=1, any single element P_i forms a sequence of 1 consecutive integer.
        # i_K - i_1 = i_1 - i_1 = 0. This is the minimum possible.
        print(0)
        return

    # pos[value] stores the 0-indexed actual position of 'value' in P_list.
    # P_list contains values from 1 to N.
    # pos array will be 1-indexed for 'value', so pos[0] is unused.
    pos = [0] * (N + 1) 
    for i in range(N): # i is the 0-indexed position
        value_at_P_list_i = P_list[i]
        pos[value_at_P_list_i] = i

    # min_dq stores 'value's, such that their actual positions (pos[value]) are in strictly increasing order.
    # min_dq[0] will give the 'value' that has the minimum position in the current window.
    min_dq = deque()
    # max_dq stores 'value's, such that their actual positions (pos[value]) are in strictly decreasing order.
    # max_dq[0] will give the 'value' that has the maximum position in the current window.
    max_dq = deque()
    
    min_overall_span = sys.maxsize # Stores min(i_K - i_1)

    # We iterate through 'value_being_added' from 1 to N.
    # This 'value_being_added' is the rightmost value in the current sliding window of K consecutive values.
    # For example, if K=3 and value_being_added=5, the window of values is {3, 4, 5}.
    for value_being_added in range(1, N + 1):
        current_actual_pos = pos[value_being_added] # 0-indexed actual position of value_being_added

        # Maintain min_dq:
        # Remove 'value's from the right of min_dq if their actual positions are >= current_actual_pos.
        # This ensures that pos[min_dq[0]] < pos[min_dq[1]] < ...
        while min_dq and pos[min_dq[-1]] >= current_actual_pos:
            min_dq.pop()
        min_dq.append(value_being_added)

        # Maintain max_dq:
        # Remove 'value's from the right of max_dq if their actual positions are <= current_actual_pos.
        # This ensures that pos[max_dq[0]] > pos[max_dq[1]] > ...
        while max_dq and pos[max_dq[-1]] <= current_actual_pos:
            max_dq.pop()
        max_dq.append(value_being_added)
        
        # The current window of values is [value_being_added - K + 1, ..., value_being_added].
        # 'value's that are value_being_added - K or less are out of this window.
        # Remove such 'value's from the left of deques.
        
        # value_limit_to_remove is the largest value that is no longer in the window.
        # e.g. if K=3, value_being_added=3, window is {1,2,3}. value_limit_to_remove = 3-3 = 0. Values <=0 removed.
        # e.g. if K=3, value_being_added=4, window is {2,3,4}. value_limit_to_remove = 4-3 = 1. Values <=1 removed (i.e. 1 removed).
        value_limit_to_remove = value_being_added - K
        
        while min_dq and min_dq[0] <= value_limit_to_remove:
            min_dq.popleft()
        while max_dq and max_dq[0] <= value_limit_to_remove:
            max_dq.popleft()

        # If window is full (i.e., we have processed at least K values, so value_being_added >= K)
        # The first time this condition is true is when value_being_added = K.
        # At this point, the window of values is [1, ..., K].
        if value_being_added >= K:
            # The minimum actual position in P_list for the current window of K values
            min_actual_pos_in_window = pos[min_dq[0]]
            # The maximum actual position in P_list for the current window of K values
            max_actual_pos_in_window = pos[max_dq[0]]
            
            current_span = max_actual_pos_in_window - min_actual_pos_in_window
            if current_span < min_overall_span:
                min_overall_span = current_span
                
    print(min_overall_span)

if __name__ == '__main__':
    main()