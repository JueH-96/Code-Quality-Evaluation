import bisect

# Read the number of sleep log entries and the sleep log itself
N = int(input())
A = list(map(int, input().split()))

# Read the number of questions
Q = int(input())

# Process each question
for _ in range(Q):
    l, r = map(int, input().split())
    
    # Find the index of the first sleep time that is greater than l
    start_index = bisect.bisect_left(A, l)
    if start_index % 2 == 0:
        start_index -= 1
    
    # Find the index of the last sleep time that is less than or equal to r
    end_index = bisect.bisect_right(A, r)
    if end_index % 2 == 0:
        end_index -= 1
    
    # Calculate the total sleep time within the given range
    total_sleep_time = 0
    for i in range(start_index, end_index, 2):
        sleep_start = max(l, A[i])
        sleep_end = min(r, A[i+1])
        total_sleep_time += sleep_end - sleep_start
    
    # Output the total sleep time for the current question
    print(total_sleep_time)