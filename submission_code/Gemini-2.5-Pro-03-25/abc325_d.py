# YOUR CODE HERE
import heapq
import sys

# Function to solve the problem
def solve():
    N = int(sys.stdin.readline())
    
    # Handle edge case N=0
    if N == 0:
        print(0)
        return

    products = []
    # Read product data and store it along with calculated end times
    # Uses 0-based indexing internally for product IDs
    for i in range(N):
        T, D = map(int, sys.stdin.readline().split())
        # Product ID is i (0-based index)
        # The problem description and sample explanation implies the interval is inclusive: [T, T+D].
        # The product enters at time T and leaves *after* T+D.
        # So it is available for printing at any time t such that T <= t <= T+D.
        products.append({'id': i, 'T': T, 'D': D, 'end': T + D})

    events = []
    # Create enter and leave events for each product
    for i in range(N):
        # Enter event: (time, type=0, product_id)
        events.append((products[i]['T'], 0, i)) 
        # Leave event: (time = end_time + 1, type=1, product_id)
        # The product is available up to and including time 'end' (T+D). 
        # It is definitely gone at time 'end + 1'. This time marks when it's no longer available.
        events.append((products[i]['end'] + 1, 1, i)) 
    
    # Sort events. Primary key: time. Secondary key: type (0=enter before 1=leave).
    # This ensures correct processing order when events occur at the same time.
    # For example, if a product enters at the same time another leaves, we process the enter first.
    events.sort()

    # Min-heap (priority queue) to store available products, ordered by their end time.
    # Stores tuples: (end_time, product_id)
    available_pq = [] 
    
    # Set to keep track of product IDs whose 'leave' event has occurred OR have expired
    # before they could be printed. This is used to filter out items from PQ or prevent adding items.
    # Using a set allows O(1) average time complexity for checks.
    left_ids = set() 
    
    printed_count = 0 # Counter for the number of products printed
    # Time when the last print operation finished. Printer is ready at time + 1.
    # Initialize to a value safely before any possible print time (e.g., -1 if times are non-negative).
    last_print_time = -1 
    
    event_idx = 0 # Index for iterating through the sorted events list
    
    # Get unique event times to iterate through. This handles large time gaps efficiently
    # by jumping the simulation time between relevant event points.
    unique_times = sorted(list(set(e[0] for e in events)))
    
    # If there are no events (N=0 was handled, but check just in case), exit.
    if not unique_times:
        print(0) 
        return

    # Start simulation time at the time of the first event
    current_sim_time = unique_times[0] 

    # Iterate through each unique event time point `t`
    for t in unique_times:
        
        # --- Process the time interval [current_sim_time, t) ---
        # In this interval, no new events occur. We attempt to print available products using the greedy strategy.
        while True:
            # Calculate the earliest time the printer is ready after the last print.
            ready_time = last_print_time + 1
            # The effective time for the next potential print is the maximum of
            # current simulation time and when the printer is ready. We cannot print before the simulation progresses to `current_sim_time`.
            effective_time = max(current_sim_time, ready_time)

            # If the next possible print time is at or after the current event time 't',
            # we cannot print anything more before this event occurs. Advance simulation time to 't' and break this inner loop.
            if effective_time >= t:
                current_sim_time = t
                break
            
            # Clean the top of the priority queue: remove products whose 'leave' event has already occurred
            # or were marked as expired previously. These products are no longer candidates for printing.
            while available_pq and available_pq[0][1] in left_ids:
                 heapq.heappop(available_pq)

            # If there are no available products currently in the PQ after cleanup, 
            # advance simulation time to 't' and break the inner loop. Nothing to print.
            if not available_pq:
                current_sim_time = t
                break

            # Peek at the product with the earliest end time. This is our candidate for the next print according to the greedy strategy.
            earliest_end_time, pid = available_pq[0]

            # Check if this candidate product expires before the printer is ready (effective_time).
            # A product i must be printed at time p_time such that T_i <= p_time <= T_i + D_i (which is 'end').
            # If the earliest we can print (`effective_time`) is after its end time (`earliest_end_time`), it's too late.
            if effective_time > earliest_end_time:
                # The product is no longer available by the time we could print it. Remove it from PQ.
                heapq.heappop(available_pq) 
                # Mark it as effectively 'left' or expired. This prevents re-adding it and ensures future cleanup.
                left_ids.add(pid) 
                # The simulation time must reflect that we considered up to 'effective_time'.
                # We might be able to print another item starting from this time.
                current_sim_time = effective_time 
                continue # Continue the loop to check the next product in the PQ

            # If we reach here, product 'pid' is available, and the printer is ready at 'effective_time'.
            # We choose to print this product based on the greedy strategy (earliest end time).
            heapq.heappop(available_pq) # Remove the printed product from PQ
            
            printed_count += 1 # Increment print count
            last_print_time = effective_time # Update the time of the last print operation
            
            # Simulation time advances. The printer will be busy until effective_time + 1.
            # The simulation state effectively moves forward to this point.
            current_sim_time = effective_time + 1 
            
            # Check if after printing, the simulation time reaches or exceeds the next event time 't'.
            # If so, cap the simulation time at 't' and stop printing for this interval.
            if current_sim_time >= t:
                 current_sim_time = t 
                 break
                
        # --- Process events occurring exactly AT time 't' ---
        # Ensure simulation time is exactly 't' before processing events scheduled for this time.
        current_sim_time = t 
        # Process all events that occur at time 't'
        while event_idx < len(events) and events[event_idx][0] == t:
            evt_time, evt_type, pid = events[event_idx]
            
            if evt_type == 0: # Product 'pid' enters the range
                # Add product to the priority queue only if its leave event hasn't happened yet
                # AND it wasn't previously marked as expired/left.
                if pid not in left_ids:
                     # Push tuple (end_time, product_id) onto the min-heap.
                     heapq.heappush(available_pq, (products[pid]['end'], pid))
            else: # Product 'pid' leaves the range (event time is end_time + 1)
                # Mark this product ID as having left. Any instance of this pid in the PQ 
                # will be filtered out upon extraction (during the cleanup step above).
                left_ids.add(pid)
            
            event_idx += 1 # Move to the next event in the sorted list
        
        # After processing events at time 't', the simulation time `current_sim_time` is correctly set to `t`.
        # This prepares the state for processing the next time interval `[t, next_t)`.

    # Finally, print the total number of products printed.
    print(printed_count)

# Ensure the solve function is called only when the script is executed directly
if __name__ == '__main__':
    solve()