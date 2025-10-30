import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums
    # prefix_sum[i] stores sum of A[0]...A[i-1]
    # prefix_sum has N+1 elements, prefix_sum[0] = 0
    prefix_sum = [0] * (N + 1)
    current_running_sum = 0
    for i in range(N):
        current_running_sum += A[i]
        prefix_sum[i+1] = current_running_sum

    # Find all divisors of N
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i * i != N: # Avoid adding the square root twice if N is a perfect square
                divisors.append(N // i)
    
    max_overall_difference = 0
    
    # Iterate over each possible k (number of boxes per truck)
    # k must be a divisor of N
    for k_boxes_per_truck in divisors:
        num_trucks = N // k_boxes_per_truck
        
        # If there's only one truck (i.e., k_boxes_per_truck = N), the difference is 0.
        # This case is handled correctly by the general logic below:
        # min_truck_weight_for_this_k and max_truck_weight_for_this_k 
        # will both be the total sum of all boxes, so their difference is 0.
        # max_overall_difference is initialized to 0 and only increases.
        
        min_truck_weight_for_this_k = float('inf')
        # Weights a_i are positive, so sums are positive. float('-inf') is robust.
        max_truck_weight_for_this_k = float('-inf') 

        for j_truck_idx in range(num_trucks): # j_truck_idx is the 0-indexed truck number
            # Truck j_truck_idx gets boxes from index j_truck_idx*k_boxes_per_truck 
            # to (j_truck_idx+1)*k_boxes_per_truck - 1 (0-indexed in A)
            
            # Using prefix sums:
            # Sum of A[s_idx] ... A[e_idx] = prefix_sum[e_idx+1] - prefix_sum[s_idx]
            # Here, s_idx = j_truck_idx * k_boxes_per_truck
            # e_idx = (j_truck_idx + 1) * k_boxes_per_truck - 1
            # So, sum = prefix_sum[ (j_truck_idx+1)*k_boxes_per_truck ] - prefix_sum[ j_truck_idx*k_boxes_per_truck ]
            
            end_idx_in_prefix_sum_array = (j_truck_idx + 1) * k_boxes_per_truck
            start_idx_in_prefix_sum_array = j_truck_idx * k_boxes_per_truck
            
            current_truck_weight = prefix_sum[end_idx_in_prefix_sum_array] - prefix_sum[start_idx_in_prefix_sum_array]
            
            if current_truck_weight < min_truck_weight_for_this_k:
                min_truck_weight_for_this_k = current_truck_weight
            if current_truck_weight > max_truck_weight_for_this_k:
                max_truck_weight_for_this_k = current_truck_weight
                
        current_difference = max_truck_weight_for_this_k - min_truck_weight_for_this_k
        
        if current_difference > max_overall_difference:
            max_overall_difference = current_difference
            
    sys.stdout.write(str(max_overall_difference) + "
")

# Read number of test cases
T = int(sys.stdin.readline())
for _ in range(T):
    solve()