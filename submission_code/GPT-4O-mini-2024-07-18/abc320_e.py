def flowing_noodles():
    import sys
    from collections import deque

    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Initialize the noodle counts and the queue for people
    noodle_counts = [0] * N
    queue = deque(range(N))  # Queue of people indexed from 0 to N-1
    return_times = [0] * N  # When each person will return to the queue
    
    # Process each event
    for i in range(1, M + 1):
        T_i, W_i, S_i = map(int, data[i].split())
        
        # Handle the return of people who should return before or at T_i
        for j in range(N):
            if return_times[j] <= T_i:
                if j in queue:
                    queue.remove(j)
        
        # If there is someone in the queue, give them the noodles
        if queue:
            front_person = queue.popleft()
            noodle_counts[front_person] += W_i
            return_times[front_person] = T_i + S_i
            queue.append(front_person)  # They go to the back of the queue
    
    # Output the results
    for count in noodle_counts:
        print(count)