def solve():
    N, D, P = map(int, input().split())
    F_list = list(map(int, input().split()))

    F_list.sort()

    # prefix_sum[k] will store the sum of the k smallest fares.
    # prefix_sum[0] = 0
    # prefix_sum[i] stores F_list[0] + ... + F_list[i-1]
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + F_list[i]

    # Cost if 0 batches are bought (pay all regular fares)
    min_total_cost = prefix_sum[N] 

    # Maximum number of batches we might consider. 
    # If N=5, D=2, max_batches_to_consider = ceil(5/2) = 3.
    # Buying more batches than this would give more passes than needed to cover all N days,
    # increasing pass costs without reducing fare costs further (which would already be 0).
    max_batches_to_consider = (N + D - 1) // D 
    
    # Iterate through number of batches purchased, from 1 up to max_batches_to_consider
    for k_batches in range(1, max_batches_to_consider + 1):
        cost_of_passes = k_batches * P
        
        # Passes obtained from k_batches
        num_passes_obtained = k_batches * D
        
        # We use passes on the most expensive days.
        # The number of days actually covered by passes is limited by N.
        num_days_covered_by_pass = min(num_passes_obtained, N)
        
        # The remaining days must be paid with regular fare.
        # These are the (N - num_days_covered_by_pass) cheapest days.
        num_days_to_pay_regular_fare = N - num_days_covered_by_pass
        
        # Cost of these regular fares is the sum of the smallest num_days_to_pay_regular_fare fares.
        cost_of_regular_fares = prefix_sum[num_days_to_pay_regular_fare]
        
        current_total_cost = cost_of_passes + cost_of_regular_fares
        
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            
    print(min_total_cost)

if __name__ == '__main__':
    solve()