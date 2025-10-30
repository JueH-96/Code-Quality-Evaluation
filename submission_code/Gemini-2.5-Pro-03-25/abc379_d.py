# YOUR CODE HERE
import sys
import heapq

def solve():
    """
    Solves the plant growing and harvesting problem using a min-heap.
    
    The core idea is to track each plant not by its current height, but by the value
    of a global 'total_growth' counter at the time the plant was added. Let this value
    be 'plant_time_growth'. The global 'total_growth' counter increases whenever 
    a type 2 query (wait T days) occurs.
    
    The current height of a plant can then be calculated as:
    current_height = current_total_growth - plant_time_growth
    
    This approach avoids updating the height of every plant during type 2 queries.
    Instead, only the global 'total_growth' is updated.
    
    For type 3 queries (harvest plants with height >= H), the condition becomes:
    current_total_growth - plant_time_growth >= H
    Rearranging this gives the condition on 'plant_time_growth':
    plant_time_growth <= current_total_growth - H
    
    We use a min-heap to store the 'plant_time_growth' values of all active plants.
    This allows efficient retrieval and removal of the plant with the smallest 
    'plant_time_growth', which corresponds to the plant that was planted earliest
    relative to the accumulated growth.
    
    In a type 3 query, we repeatedly check the minimum element in the heap. If its 
    'plant_time_growth' satisfies the derived condition (<= threshold), we remove it 
    (harvest it) and increment a counter. This continues until the minimum element 
    does not satisfy the condition or the heap becomes empty.
    """
    
    # Read the number of queries from standard input
    Q = int(sys.stdin.readline())
    
    # Initialize a min-heap (implemented using a list in Python's heapq module)
    # This heap will store the 'plant_time_growth' values for each plant currently growing.
    plants_heap = [] 
    
    # Initialize the total accumulated growth counter. This starts at 0 and increases
    # with each type 2 query.
    total_growth = 0
    
    # Process each of the Q queries
    for _ in range(Q):
        # Read the query line and split it into components based on spaces
        query_line = sys.stdin.readline().split()
        # The first component is always the query type
        query_type = int(query_line[0])
        
        if query_type == 1:
            # Type 1 Query: Plant a new plant.
            # The plant starts with height 0 at this moment.
            # We record the current 'total_growth' value and associate it with this plant.
            # This 'plant_time_growth' value is pushed onto the min-heap.
            heapq.heappush(plants_heap, total_growth)
            
        elif query_type == 2:
            # Type 2 Query: Wait T days.
            # All existing plants grow by T height units.
            # This is modeled by increasing the global 'total_growth' counter by T.
            # The individual plant records ('plant_time_growth') are not modified.
            T = int(query_line[1])
            total_growth += T
            
        elif query_type == 3:
            # Type 3 Query: Harvest plants meeting a height requirement.
            # Harvest all plants with current height >= H.
            H = int(query_line[1])
            
            # Calculate the threshold based on the harvest condition derived earlier:
            # plant_time_growth <= current_total_growth - H
            threshold = total_growth - H
            
            # Initialize a counter for the number of plants harvested in this query.
            harvested_count = 0
            
            # Check the plant with the minimum 'plant_time_growth'. This is the root of the min-heap.
            # The condition `plants_heap` checks if the heap is not empty.
            # `plants_heap[0]` accesses the minimum element without removing it.
            # Loop as long as the heap is not empty and the minimum element satisfies the harvest condition.
            while plants_heap and plants_heap[0] <= threshold:
                # Remove the minimum element (plant) from the heap using heappop. This simulates harvesting.
                heapq.heappop(plants_heap)
                # Increment the count of harvested plants for this specific query.
                harvested_count += 1
            
            # After checking and potentially removing all eligible plants, print the total count
            # of plants harvested in response to this type 3 query.
            print(harvested_count)

# Execute the main function to start the simulation process based on standard input.
solve()