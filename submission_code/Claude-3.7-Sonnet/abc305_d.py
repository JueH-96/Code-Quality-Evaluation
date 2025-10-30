N = int(input())
A = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    l, r = map(int, input().split())
    sleep_time = 0
    for i in range(1, (N-1)//2 + 1):
        bed_time = A[2*i]
        wake_time = A[2*i + 1]
        
        # If the sleep session is completely after the query range, break
        if bed_time >= r:
            break
        
        # If the sleep session is completely before the query range, skip
        if wake_time <= l:
            continue
        
        # Compute the overlap
        overlap_start = max(l, bed_time)
        overlap_end = min(r, wake_time)
        
        sleep_time += overlap_end - overlap_start
    
    print(sleep_time)