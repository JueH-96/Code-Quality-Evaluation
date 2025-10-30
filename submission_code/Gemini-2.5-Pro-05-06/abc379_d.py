import sys
import heapq

# Use sys.stdin.readline for faster input, especially for large Q
input = sys.stdin.readline

def main():
    Q = int(input())

    # min_heap_sbirth stores the 'current_growth_sum' value at the time each plant was planted.
    # A plant's S_birth value essentially represents its "birth time" relative to total growth.
    # Plants with smaller S_birth values are older and thus, generally, taller.
    min_heap_sbirth = []
    
    # current_growth_sum tracks the total amount of growth all plants have experienced
    # due to type 2 queries (waiting T days). It starts at 0.
    current_growth_sum = 0

    # Process each query
    for _ in range(Q):
        query_line_parts = input().split() # Read the query line and split into parts
        query_type = int(query_line_parts[0]) # The first part is the query type

        if query_type == 1:
            # Type 1: Plant a new plant.
            # Its initial height is 0.
            # We record its S_birth, which is the current_growth_sum at this moment.
            # This S_birth value is pushed onto the min-heap.
            heapq.heappush(min_heap_sbirth, current_growth_sum)
        
        elif query_type == 2:
            # Type 2: Wait for T days.
            # The height of every existing plant increases by T.
            # This is modeled by adding T to current_growth_sum.
            T = int(query_line_parts[1])
            current_growth_sum += T
            
        elif query_type == 3:
            # Type 3: Harvest all plants with a height of at least H.
            H = int(query_line_parts[1])
            
            # A plant's current height is: current_growth_sum - S_birth
            # (S_birth is the value stored in the heap for that plant).
            # We need to harvest if: (current_growth_sum - S_birth) >= H
            # Rearranging this gives a condition on S_birth:
            # S_birth <= current_growth_sum - H
            # Let's call this the 'harvest_threshold_for_sbirth'.
            harvest_threshold_for_sbirth = current_growth_sum - H
            
            harvested_count = 0
            # We check plants in the min-heap. The min-heap gives plants
            # with the smallest S_birth values first. These are the oldest plants
            # and consequently (for a given current_growth_sum) the tallest ones.
            # If a plant's S_birth is less than or equal to harvest_threshold_for_sbirth,
            # it means its height is >= H, so it's harvested.
            # The loop continues as long as the heap is not empty AND
            # the S_birth of the oldest plant (min_heap_sbirth[0]) meets the condition.
            while min_heap_sbirth and min_heap_sbirth[0] <= harvest_threshold_for_sbirth:
                heapq.heappop(min_heap_sbirth) # Remove this plant from the heap
                harvested_count += 1           # Increment count of harvested plants
            
            # Output the number of plants harvested in this query
            print(harvested_count)

# Standard Python construct to run main() when the script is executed
if __name__ == '__main__':
    main()