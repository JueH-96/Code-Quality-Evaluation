# YOUR CODE HERE
import heapq
import sys

# Use fast I/O
# sys.stdin.readline is generally faster than input()
# for reading lines in competitive programming
input = sys.stdin.readline

def solve():
    # Read the total number of queries
    Q = int(input())

    # Min-heap to store the 'total_height_increase' value at the time
    # each plant was planted. This value represents the base height
    # accumulated from type 2 queries *before* planting.
    # The current height of a plant is (current_total_height_increase - planting_time_increase_at_planting).
    plants_planting_time_increase = []

    # Variable to track the cumulative height increase from all Type 2 queries processed so far.
    # Using Python's arbitrary precision integers automatically handles large values.
    total_height_increase = 0

    # Process each query
    for _ in range(Q):
        # Read the query line and split it into integers
        # The first element is the query type. For type 2 and 3, the second element is the parameter.
        query = list(map(int, input().split()))
        query_type = query[0]

        if query_type == 1:
            # Type 1: Prepare one empty flower pot and put a plant in it.
            # Initial height is 0.
            # When a plant is added, its "base level" for future height increases is the
            # total increase accumulated *up to this point*. We store this value.
            # A plant planted now effectively starts at height 0 relative to the
            # current accumulated increase. Its future height will be (future_total_increase - current_total_increase).
            heapq.heappush(plants_planting_time_increase, total_height_increase)

        elif query_type == 2:
            # Type 2 T: Wait for T days. Height of every existing plant increases by T.
            # This uniform increase across all plants is accounted for by incrementing
            # the global cumulative increase variable.
            T = query[1]
            total_height_increase += T

        elif query_type == 3:
            # Type 3 H: Harvest plants with height >= H, and output the number harvested.
            H = query[1]
            harvested_count = 0

            # For a plant planted when the cumulative increase was S_plant, its current height is:
            # (current_total_height_increase) - (S_plant)
            # We need to harvest plants where this current height is >= H:
            # current_total_height_increase - S_plant >= H
            # Rearranging the inequality to find the condition on S_plant:
            # S_plant <= current_total_height_increase - H
            # This gives us the threshold value for the planting_time_increase stored in the heap.
            threshold = total_height_increase - H

            # The min-heap 'plants_planting_time_increase' is ordered by the S_plant values.
            # The plants with the smallest S_plant values have received the largest
            # total height increase since they were planted (current_total_height_increase - S_plant).
            # These are the tallest plants (relative to the zero-height start).
            # We efficiently remove plants whose S_plant value is <= threshold by
            # repeatedly popping from the min-heap's root as long as the condition holds.
            while plants_planting_time_increase and plants_planting_time_increase[0] <= threshold:
                heapq.heappop(plants_planting_time_increase)
                harvested_count += 1

            # Output the number of plants harvested in this specific Type 3 query.
            print(harvested_count)

# Execute the main simulation function
solve()