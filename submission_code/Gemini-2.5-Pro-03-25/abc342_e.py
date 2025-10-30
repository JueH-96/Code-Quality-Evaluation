import sys
import heapq

# Function to read input faster
def read_ints():
    """Reads a line of space-separated integers from stdin."""
    return map(int, sys.stdin.readline().split())

def solve():
    """Solves the train scheduling problem."""
    N, M = read_ints()

    # Store incoming train information for each station
    # incoming_trains[b] will store list of tuples (l, d, k, c, a) for trains arriving at station b from station a.
    # Stations are 1-indexed, so we use lists/arrays of size N+1.
    incoming_trains = [[] for _ in range(N + 1)]
    for _ in range(M):
        # Read train information: l_i, d_i, k_i, c_i, A_i, B_i
        # l: first departure time
        # d: interval between departures
        # k: number of trains in the series
        # c: travel time
        # a: departure station (A_i)
        # b: arrival station (B_i)
        l, d, k, c, a, b = read_ints()
        # Store the tuple (l, d, k, c, a) associated with the destination station b.
        # This represents an edge in the reversed graph perspective used by the algorithm.
        incoming_trains[b].append((l, d, k, c, a))

    # Initialize latest departure times for each station
    # We use -1 to represent negative infinity, indicating that initially, we don't know a path
    # from any station (except N) to N.
    latest_departure = [-1] * (N + 1) 
    
    # Use a very large integer to represent infinity for the destination station N.
    # The maximum possible time could be around l_i + (k_i-1)*d_i + c_i.
    # With values up to 10^9, this can reach approximately 10^18.
    # Let's pick a value safely larger than any possible finite time.
    INF = 2 * 10**18 + 10**9 + 7 
    # The latest departure time from station N to reach N is considered infinite.
    latest_departure[N] = INF

    # Max-priority queue storing (-time, station) pairs.
    # We use negative time because Python's heapq module implements a min-heap.
    # By negating the times, the element with the largest time will have the smallest negative value,
    # effectively making it a max-heap based on time.
    pq = [(-INF, N)] # Start the Dijkstra-like algorithm from the destination station N.

    # Main loop of the Dijkstra-like algorithm
    while pq:
        # Extract the station u with the current maximum latest departure time found so far.
        neg_time, u = heapq.heappop(pq)
        current_max_time_at_u = -neg_time # Revert negation to get the actual time

        # Optimization: If the extracted time is less than the already known best (latest) departure time for station u,
        # this means we found a better path to u earlier and this queue entry is outdated. Skip it.
        if current_max_time_at_u < latest_departure[u]:
             continue

        # Explore all trains arriving at station u. These correspond to edges leading into u in the original graph,
        # or edges outgoing from u in our reversed graph perspective.
        for l, d, k, c, v in incoming_trains[u]:
            # Information for a train series from station v to station u:
            # l: first departure time from v
            # d: interval between departures
            # k: number of trains in the series
            # c: travel time from v to u
            
            # 'current_max_time_at_u' represents the latest time we must arrive at station u
            # to be able to continue on a path towards station N departing at `latest_departure[u]`.
            
            # We need to find the latest possible departure time 't' from station v using this train series,
            # such that the arrival time 't + c' at u is less than or equal to 'current_max_time_at_u'.
            
            # First, check if even the earliest train in this series (departing at l) arrives too late.
            # The earliest arrival time at u is l + c.
            if l + c > current_max_time_at_u:
                # If the first train arrives later than required, all subsequent trains in the series
                # will also arrive too late. So this train series cannot be used to reach u by the required time.
                continue 

            # Now, we need to find the maximum index j_star (where 0 <= j_star <= k-1)
            # such that the train departing at time t = l + j_star * d satisfies the arrival time constraint:
            # t + c <= current_max_time_at_u
            
            # Substitute t: l + j_star * d + c <= current_max_time_at_u
            # Rearrange: l + j_star * d <= current_max_time_at_u - c
            # Further rearrange: j_star * d <= current_max_time_at_u - c - l
            
            # Handle the case where the required arrival time at u is INF (initially only for N)
            if current_max_time_at_u == INF:
                # If the allowed arrival time at u is effectively infinity, any train in the series works time-wise.
                # To maximize the departure time from v, we should take the latest possible train in this series.
                # The latest train corresponds to the maximum index j = k-1.
                 j_star = k - 1
            else:
                # 'current_max_time_at_u' is a finite integer value.
                # Let R represent the deadline for the value `l + j*d`: R_limit = current_max_time_at_u - c
                # Constraint becomes: l + j*d <= R_limit
                # Isolate the term with j: j*d <= R_limit - l
                # Let R = R_limit - l = current_max_time_at_u - c - l.
                # We already checked that l + c <= current_max_time_at_u, which implies R >= 0.
                R = current_max_time_at_u - c - l
                
                # Find the maximum integer j such that j*d <= R. Since d >= 1, d is positive.
                # Divide by d: j <= R / d.
                # The maximum integer j satisfying this is floor(R / d).
                # In Python, integer division // performs floor division.
                max_j_allowed_by_time = R // d 
                
                # The index j must also be within the valid range [0, k-1] for this train series.
                # So, the actual maximum valid index j_star is the minimum of the series limit (k-1)
                # and the time constraint limit (max_j_allowed_by_time).
                j_star = min(k - 1, max_j_allowed_by_time)
            
            # At this point, j_star is the maximum valid index (0 <= j_star <= k-1) for the train series
            # satisfying the arrival time constraint at u. We know j_star >= 0.
            
            # Calculate the corresponding latest possible departure time from station v using this train series.
            t_star = l + j_star * d

            # If this calculated departure time t_star from v is later than the currently known
            # latest departure time for station v (`latest_departure[v]`), it means we have found
            # a potentially better sequence of trains enabling a later start from v to reach N.
            if t_star > latest_departure[v]:
                # Update the latest departure time recorded for station v.
                latest_departure[v] = t_star
                # Add/update station v in the priority queue with its new, later departure time.
                # Remember to negate the time for the min-heap simulation of a max-heap.
                heapq.heappush(pq, (-t_star, v))

    # After the algorithm finishes, latest_departure[i] contains the maximum possible
    # departure time from station i such that it's possible to reach station N.
    # According to the problem statement re-interpretation based on samples, this value is f(i).
    
    # Prepare the output lines.
    output_lines = []
    for i in range(1, N): # Iterate through stations 1 to N-1
        if latest_departure[i] == -1:
            # If latest_departure[i] is still -1, it means station N is unreachable from station i.
            output_lines.append("Unreachable")
        else:
            # Otherwise, print the calculated latest departure time.
            output_lines.append(str(latest_departure[i]))
    
    # Write all output lines to stdout at once for potentially faster IO.
    # Ensure each line ends with a newline character.
    sys.stdout.write("
".join(output_lines) + "
")

# Execute the main function to solve the problem.
solve()