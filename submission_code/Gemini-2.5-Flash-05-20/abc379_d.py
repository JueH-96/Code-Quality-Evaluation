import heapq
import sys

def solve():
    # Read the number of queries
    Q = int(sys.stdin.readline())

    # min_heap will store the 'base_height' of plants.
    # The 'base_height' of a plant is the value of current_global_growth_offset
    # at the moment the plant was created.
    # A plant's actual height at any time is (current_global_growth_offset - its_base_height).
    min_heap = [] 
    
    # current_global_growth_offset accumulates the total height increase
    # applied to all plants since the simulation started.
    current_global_growth_offset = 0

    for _ in range(Q):
        # Read the query and parse its parts
        query_parts = list(map(int, sys.stdin.readline().split()))
        query_type = query_parts[0]

        if query_type == 1:
            # Type 1: Prepare one empty flower pot and put a plant in it.
            # The plant's initial height is 0.
            # We record its base_height, which is the current global offset.
            # Its actual height will be (current_global_growth_offset - this_base_height).
            heapq.heappush(min_heap, current_global_growth_offset)
        elif query_type == 2:
            # Type 2 T: Wait for T days.
            # The height of every existing plant increases by T.
            # This is handled by simply updating the global growth offset.
            T = query_parts[1]
            current_global_growth_offset += T
        else: # query_type == 3
            # Type 3 H: Harvest all plants with a height of at least H.
            # Output the number of plants harvested.
            H = query_parts[1]
            
            # A plant with 'base_height' X has current height
            # (current_global_growth_offset - X).
            # We want to harvest plants where (current_global_growth_offset - X >= H).
            # Rearranging this inequality to solve for X:
            # -X >= H - current_global_growth_offset
            # X <= current_global_growth_offset - H
            # So, we need to harvest all plants whose base_height is less than or equal to
            # (current_global_growth_offset - H).
            threshold_X = current_global_growth_offset - H
            
            harvested_count = 0
            # Use the min_heap to efficiently find and remove plants.
            # Since the min_heap stores base_heights, and we want to remove
            # base_heights <= threshold_X, we can repeatedly pop the smallest
            # element from the heap as long as it satisfies the condition.
            while min_heap and min_heap[0] <= threshold_X:
                heapq.heappop(min_heap)
                harvested_count += 1
            
            # Print the count of harvested plants.
            sys.stdout.write(str(harvested_count) + "
")

# Call the solve function to run the program
solve()