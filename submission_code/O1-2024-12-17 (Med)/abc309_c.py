def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    
    # Read the a_i, b_i pairs
    arr = []
    idx = 2
    for _ in range(N):
        a_i = int(input_data[idx]); b_i = int(input_data[idx+1])
        idx += 2
        arr.append((a_i, b_i))
    
    # Sort by a_i ascending
    arr.sort(key=lambda x: x[0])
    
    # Build suffix sums of b_i
    # suffixSum[i] = sum of b_j for j >= i
    suffixSum = [0]*(N+1)
    for i in range(N-1, -1, -1):
        suffixSum[i] = suffixSum[i+1] + arr[i][1]
    
    # Build a sorted list of distinct a_i plus day=1 (to check if day=1 is valid)
    distinct_days = set()
    distinct_days.add(1)
    for a_i, _ in arr:
        distinct_days.add(a_i)
    distinct_days = sorted(distinct_days)
    
    import bisect
    
    # Function to get total pills on day d
    # = sum of b_i for all i with a_i >= d
    # We find the first index idx where arr[idx].a >= d
    # Then sum is suffixSum[idx]
    def total_pills(day):
        # Binary search for the first index i where arr[i].a >= day
        i = bisect.bisect_left(arr, (day, -1))  # compare by first element
        return suffixSum[i]
    
    # Check each distinct day in ascending order
    for d in distinct_days:
        if total_pills(d) <= K:
            print(d)
            return
    
    # If none of these days worked, the answer is max(a_i) + 1
    # because from that day onward, no medicines remain (sum of pills is 0)
    print(arr[-1][0] + 1)

# Do not forget to call main()
main()