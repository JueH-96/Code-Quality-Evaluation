def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    N = int(data[index])
    index += 1
    
    A = list(map(int, data[index:index+N]))
    index += N
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        l = int(data[index])
        r = int(data[index+1])
        queries.append((l, r))
        index += 2
    
    # To handle the queries efficiently, we can use a prefix sum array that keeps track of total sleep time up to each minute.
    # We will create a large enough array to cover up to A_N minutes.
    
    # Since A_N can be very large (up to 10^9), we cannot create an array of that size.
    # Instead, we will use a map (or dictionary) to store changes in sleep status at specific times.
    
    sleep_changes = {}
    
    # Populate the sleep changes based on sleep intervals
    for i in range(1, N, 2):
        start_sleep = A[i]
        end_sleep = A[i+1]
        
        if start_sleep not in sleep_changes:
            sleep_changes[start_sleep] = 0
        if end_sleep not in sleep_changes:
            sleep_changes[end_sleep] = 0
        
        sleep_changes[start_sleep] += 1
        sleep_changes[end_sleep] -= 1
    
    # Now, we need to calculate the prefix sum of sleep times
    # We will sort the keys of sleep_changes to process in order of time
    sorted_times = sorted(sleep_changes.keys())
    
    # Prefix sum of sleep times
    current_sleeping = 0
    last_time = 0
    sleep_prefix_sum = {}
    
    for time in sorted_times:
        if last_time not in sleep_prefix_sum:
            sleep_prefix_sum[last_time] = 0
        sleep_prefix_sum[time] = sleep_prefix_sum[last_time] + (time - last_time) * current_sleeping
        
        # Update the current sleeping status
        current_sleeping += sleep_changes[time]
        last_time = time
    
    # Answer the queries
    results = []
    for l, r in queries:
        if r not in sleep_prefix_sum:
            sleep_prefix_sum[r] = sleep_prefix_sum[last_time] + (r - last_time) * current_sleeping
        if l not in sleep_prefix_sum:
            sleep_prefix_sum[l] = sleep_prefix_sum[last_time] + (l - last_time) * current_sleeping
        
        sleep_time = sleep_prefix_sum[r] - sleep_prefix_sum[l]
        results.append(sleep_time)
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()