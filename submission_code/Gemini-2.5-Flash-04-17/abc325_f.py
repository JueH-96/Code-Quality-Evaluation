import sys
import math
import heapq

# Use integer arithmetic for ceiling for positive numbers
def req_x2(D, x1, L1, L2):
    """
    Calculates the minimum number of type 2 sensors needed for a section of length D,
    given x1 type 1 sensors are already used.
    """
    remaining_D = D - x1 * L1
    if remaining_D <= 0:
        return 0
    # Use integer division equivalent for ceiling for positive remaining_D
    # (a + b - 1) // b for ceil(a/b) where a > 0, b > 0
    return (remaining_D + L2 - 1) // L2

def solve():
    N = int(sys.stdin.readline())
    D = list(map(int, sys.stdin.readline().split()))
    L1, C1, K1 = map(int, sys.stdin.readline().split())
    L2, C2, K2 = map(int, sys.stdin.readline().split())

    min_total_cost = float('inf')

    # Iterate through all possible total number of type 1 sensors used
    # num1_total can range from 0 up to K1
    for num1_total in range(K1 + 1):

        # Greedy approach to minimize the total required type 2 sensors
        # for a fixed total number of type 1 sensors (num1_total).

        # Initial state: 0 type 1 sensors used for each section.
        # Calculate the total number of type 2 sensors required initially.
        current_total_req_x2 = 0
        for d in D:
            current_total_req_x2 += req_x2(d, 0, L1, L2)

        # Priority queue to store potential improvements (reduction in total type 2 sensors).
        # Stores tuples: (-benefit, section_index)
        # The benefit is the reduction in required type 2 sensors gained by adding
        # the *next* type 1 sensor to the specified section.
        pq = []

        # x1_counts_per_section tracks how many type 1 sensors are assigned to each section *so far*
        # during the process of distributing num1_total sensors.
        x1_counts_per_section = [0] * N

        # Initialize PQ with the potential benefits of adding the 1st type 1 sensor to each section (0 -> 1).
        for i in range(N):
            # Benefit of adding the 1st sensor to section i
            benefit = req_x2(D[i], 0, L1, L2) - req_x2(D[i], 1, L1, L2)
            # Push the benefit and the section index onto the PQ.
            # We always push, even if the immediate benefit is 0, because adding this sensor
            # might enable positive benefits from subsequent sensors for this section.
            heapq.heappush(pq, (-benefit, i))

        # Greedily add num1_total type 1 sensors by picking the assignment that yields the most benefit.
        sensors_added = 0
        # We must use exactly num1_total sensors if the PQ provides enough distinct assignments.
        # If PQ becomes empty before num1_total sensors are used, it means all remaining assignments
        # (i.e., adding more sensors to any section with its current count) yield 0 benefit.
        # In this case, the current_total_req_x2 is the minimum achievable for this num1_total,
        # and the remaining sensors are effectively added without reducing the type 2 requirement further.
        while sensors_added < num1_total:
            # If PQ is empty, it means adding any more type 1 sensors yields 0 benefit for all sections
            # at their current sensor counts. The minimum total_req_x2 is already achieved.
            if not pq:
                 break

            # Get the best available assignment (highest benefit) from the priority queue
            benefit_popped, section_idx = heapq.heappop(pq)
            benefit_popped = -benefit_popped # Convert negative benefit back to positive

            # The benefit we popped is the reduction in total_req_x2 gained by adding the
            # (x1_counts_per_section[section_idx] + 1)-th sensor to section_idx.
            # This benefit value is:
            # req_x2(D[section_idx], x1_counts_per_section[section_idx], L1, L2) - req_x2(D[section_idx], x1_counts_per_section[section_idx] + 1, L1, L2)

            # Subtract the benefit achieved by adding this sensor from the total required type 2 sensors
            current_total_req_x2 -= benefit_popped

            # Increment the count of type 1 sensors used for this section
            x1_counts_per_section[section_idx] += 1

            # Increment the total number of type 1 sensors added in this num1_total iteration
            sensors_added += 1

            # Calculate the benefit of adding the *next* sensor (the (current count + 1)-th) to this section
            next_benefit = req_x2(D[section_idx], x1_counts_per_section[section_idx], L1, L2) - req_x2(D[section_idx], x1_counts_per_section[section_idx] + 1, L1, L2)

            # Push the potential benefit of the *next* sensor for this section back onto the PQ.
            # This enables the greedy choice in the next iteration.
            heapq.heappush(pq, (-next_benefit, section_idx))


        # After distributing num1_total type 1 sensors (or fewer if PQ became empty),
        # current_total_req_x2 holds the minimum total type 2 sensors required for this num1_total.
        # Check if this minimum required number of type 2 sensors is within the available budget K2.
        if current_total_req_x2 <= K2:
            # If possible, calculate the total cost for this combination and update the minimum cost found so far.
            current_cost = num1_total * C1 + current_total_req_x2 * C2
            min_total_cost = min(min_total_cost, current_cost)

    # Output the result
    # If min_total_cost is still infinity, it means no combination of sensors within the limits could cover all sections.
    if min_total_cost == float('inf'):
        print(-1)
    else:
        # The minimum cost is an integer, print it as such.
        print(int(min_total_cost))

solve()