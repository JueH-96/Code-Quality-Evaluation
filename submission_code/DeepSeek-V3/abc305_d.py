# YOUR CODE HERE
import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    Q = int(data[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        l = int(data[idx])
        r = int(data[idx+1])
        queries.append((l, r))
        idx += 2
    
    # Precompute the sleep intervals
    sleep_intervals = []
    for i in range(1, N, 2):
        start = A[i]
        end = A[i+1]
        sleep_intervals.append((start, end))
    
    # Precompute the prefix sum of sleep durations
    prefix = [0] * (len(sleep_intervals) + 1)
    for i in range(len(sleep_intervals)):
        start, end = sleep_intervals[i]
        duration = end - start
        prefix[i+1] = prefix[i] + duration
    
    # Function to calculate the total sleep time between l and r
    def get_sleep_time(l, r):
        total = 0
        # Find the first interval that starts after l
        low = 0
        high = len(sleep_intervals) - 1
        first = -1
        while low <= high:
            mid = (low + high) // 2
            if sleep_intervals[mid][0] >= l:
                first = mid
                high = mid - 1
            else:
                low = mid + 1
        if first == -1:
            return 0
        # Find the last interval that ends before r
        low = 0
        high = len(sleep_intervals) - 1
        last = -1
        while low <= high:
            mid = (low + high) // 2
            if sleep_intervals[mid][1] <= r:
                last = mid
                low = mid + 1
            else:
                high = mid - 1
        if last == -1:
            return 0
        if first > last:
            return 0
        # Calculate the total sleep time between first and last
        total = prefix[last+1] - prefix[first]
        # Adjust for the first interval if it starts before l
        if sleep_intervals[first][0] < l:
            total -= (l - sleep_intervals[first][0])
        # Adjust for the last interval if it ends after r
        if sleep_intervals[last][1] > r:
            total -= (sleep_intervals[last][1] - r)
        return total
    
    for l, r in queries:
        print(get_sleep_time(l, r))

if __name__ == "__main__":
    main()