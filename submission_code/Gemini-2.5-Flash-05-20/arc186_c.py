import sys
import heapq

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    boxes = []
    for _ in range(N):
        V, P = map(int, sys.stdin.readline().split())
        boxes.append((V, P))

    profitable_boxes_list = []
    for V, P in boxes:
        # Only consider boxes that can yield positive profit if fully utilized.
        # This is a baseline, actual profit might be lower due to Mr. Ball's play.
        if V - P > 0:
            profitable_boxes_list.append((V, P))
            
    if not profitable_boxes_list:
        sys.stdout.write("0
")
        return
            
    # Sort profitable boxes by V_i (capacity) ascending.
    # This is crucial because Mr. Ball will target the smallest capacity boxes first.
    profitable_boxes_list.sort(key=lambda x: x[0])

    max_money_increase = 0
    
    # Case 1: Only 1 profitable box available (N' == 1)
    # As observed in Sample 2: Mr. Box buys the box, gets 1 ball, then Mr. Ball can force game end.
    if len(profitable_boxes_list) == 1:
        V, P = profitable_boxes_list[0]
        max_money_increase = max(0, 1 - P) # Only 1 ball is ever placed.
        sys.stdout.write(str(max_money_increase) + "
")
        return
    
    # Case 2: Number of profitable boxes (N') is less than or equal to M.
    # Mr. Box can dedicate a unique type to each of the N' profitable boxes.
    # Mr. Ball's strategy (from Sample 1 trace): He fills the box with the smallest capacity (V_min) among the N' chosen boxes.
    # Then for each of the other N'-1 boxes, he allows exactly 1 ball (by switching types), and then terminates.
    # So, the box with the largest V_i (last one in V-sorted list) effectively becomes the 1-ball box.
    # The other N'-1 boxes are filled to their capacity.
    
    # This scenario applies when len(profitable_boxes_list) <= M.
    # Example: N'=2, M=2 (Sample 1). Sorted: (3,1), (3,1).
    # First (3,1) (V-P = 2) filled. Second (3,1) (V-P=2) gets 1 ball.
    # Profit = (3-1) + (1-1) = 2. This matches.
    if len(profitable_boxes_list) <= M:
        current_profit = 0
        
        # Sum (V-P) for all but the last box (largest V_i)
        for i in range(len(profitable_boxes_list) - 1):
            V, P = profitable_boxes_list[i]
            current_profit += (V - P)
            
        # Add (1 - P) for the last box (largest V_i in this set)
        V_last, P_last = profitable_boxes_list[-1]
        current_profit += (1 - P_last)
        
        max_money_increase = max(0, current_profit)
        sys.stdout.write(str(max_money_increase) + "
")
        return

    # Case 3: Number of profitable boxes (N') is greater than M.
    # Mr. Box must select exactly M boxes from the N' profitable ones.
    # He wants to maximize his profit using the formula:
    # (sum of (V_j - P_j) for M-1 boxes) + (1 - P_Mth_V_smallest).
    # The `M-th_V_smallest` box refers to the one with the M-th smallest V_i among the chosen M boxes.
    # This means, Mr. Box chooses M boxes. Among these M chosen boxes, the one with the smallest capacity (V_min_selected)
    # gets filled to V_min_selected. The other M-1 boxes each get 1 ball.
    # So, total balls: V_min_selected + (M - 1).
    # Total cost: sum(P_i for the M chosen boxes).
    # Profit = (V_min_selected + M - 1) - sum(P_i for the M chosen boxes).

    # To maximize this, Mr. Box iterates through `profitable_boxes_list` (sorted by V_i).
    # Each `profitable_boxes_list[i]` is a candidate for `V_min_selected`.
    # For `V_min_selected = profitable_boxes_list[i]`, we need to pick `M-1` other boxes.
    # These `M-1` boxes must come from `profitable_boxes_list[0...i-1]` (boxes with smaller V_j)
    # and `profitable_boxes_list[i+1...N'-1]` (boxes with larger V_j).
    # From these, Mr. Box picks the ones that minimize their `P_j` values.
    
    # This is a standard sliding window maximum sum problem.
    # We maintain a min-heap `pq` of size `M-1` to store the `P_j` values of the `M-1` cheapest boxes
    # chosen from `profitable_boxes_list[0...i-1]`.
    # `current_sum_P_for_other_boxes` stores sum of P values in `pq`.
    
    current_sum_P_for_other_boxes = 0
    pq = [] # min-heap to store P_j values for the M-1 cheapest boxes

    # Populate `pq` with the `M-1` cheapest boxes from the first `M-1` elements in `profitable_boxes_list`.
    # These are the initial candidates for the `M-1` "other" boxes.
    for i in range(M - 1):
        V, P = profitable_boxes_list[i]
        heapq.heappush(pq, P)
        current_sum_P_for_other_boxes += P

    # Iterate from the `M`-th element (index `M-1`) to the end of `profitable_boxes_list`.
    # Each `profitable_boxes_list[i]` is a candidate for `V_min_selected`.
    for i in range(M - 1, len(profitable_boxes_list)):
        V_curr, P_curr = profitable_boxes_list[i]
        
        # Calculate the profit if `(V_curr, P_curr)` is the `V_min_selected` box.
        # Balls: (V_curr + M - 1). Cost: (current_sum_P_for_other_boxes + P_curr).
        current_profit = (V_curr + M - 1) - (current_sum_P_for_other_boxes + P_curr)
        max_money_increase = max(max_money_increase, current_profit)
        
        # Slide the window for the next iteration:
        # `profitable_boxes_list[i]` now becomes a candidate for one of the `M-1` "other" boxes.
        # Compare `P_curr` with the largest `P_j` currently in `pq` (which is `pq[0]`).
        # If `P_curr` is cheaper, replace the largest `P_j` in `pq` with `P_curr`.
        if i + 1 < len(profitable_boxes_list): # Check if there are elements remaining for the sliding window
            P_next_candidate = profitable_boxes_list[i + 1][1]
            if P_next_candidate < pq[0]:
                current_sum_P_for_other_boxes -= heapq.heappop(pq)
                heapq.heappush(pq, P_next_candidate)
                current_sum_P_for_other_boxes += P_next_candidate

    sys.stdout.write(str(max(0, max_money_increase)) + "
")


num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()