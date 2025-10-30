import sys
import heapq

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Sort A (boxes) in descending order.
    # A_i represents both candy count and price.
    A.sort(reverse=True)
    
    # Sort B (requirements for people) in descending order.
    # We satisfy the most demanding people first.
    B.sort(reverse=True)

    total_cost = 0
    
    # Min-heap to store prices of boxes that are currently available
    # and meet the requirement of the current person (and any previous, more demanding ones).
    # We use a min-heap because we want to pick the cheapest among eligible boxes.
    eligible_boxes_prices = [] 
    
    a_ptr = 0 # Pointer to iterate through the sorted A (boxes) list

    # Iterate through each person's requirement, from most demanding to least.
    for i in range(M):
        current_B_requirement = B[i]

        # Add all boxes from A (that are not yet considered)
        # whose candy count is sufficient for the current requirement.
        # Since A is sorted descending, we add boxes with higher candy counts first.
        while a_ptr < N and A[a_ptr] >= current_B_requirement:
            heapq.heappush(eligible_boxes_prices, A[a_ptr])
            a_ptr += 1
        
        # If the heap is empty, it means there are no available boxes
        # (among those not yet used) that can satisfy the current person's requirement.
        # Therefore, it's impossible to satisfy all conditions.
        if not eligible_boxes_prices:
            print("-1")
            return
        
        # From the eligible boxes, pick the cheapest one (smallest price)
        # by popping from the min-heap.
        chosen_box_price = heapq.heappop(eligible_boxes_prices)
        total_cost += chosen_box_price
    
    # If the loop completes, all M people have been assigned a box.
    # Print the minimum total cost.
    print(total_cost)

# Call the solve function to run the program
solve()