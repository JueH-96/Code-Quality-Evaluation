import bisect

def find_first_day():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    K = int(data[idx+1])
    idx +=2
    pairs = []
    for _ in range(N):
        a = int(data[idx])
        b = int(data[idx+1])
        pairs.append((a, b))
        idx +=2
    
    # Sort the pairs by a_i
    pairs.sort()
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    
    # Compute suffix sums
    suffix_sum = [0] * (N +1)
    for i in range(N-1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + b[i]
    
    max_a = a[-1] if N >0 else 0
    low = 1
    high = max_a +1
    answer = high  # Initialize with the maximum possible value
    
    while low <= high:
        mid = (low + high) //2
        # Find the first index where a_i >= mid
        pos = bisect.bisect_left(a, mid)
        current_sum = suffix_sum[pos]
        if current_sum <= K:
            answer = mid
            high = mid -1
        else:
            low = mid +1
    print(answer)

find_first_day()