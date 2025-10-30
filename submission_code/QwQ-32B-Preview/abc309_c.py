import sys
import bisect

def main():
    import sys
    import bisect
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    meds = []
    for i in range(N):
        a = int(data[2 + 2*i])
        b = int(data[3 + 2*i])
        meds.append((a, b))
    
    # Sort medicines by a_i in ascending order
    meds.sort(key=lambda x: x[0])
    
    # Compute suffix sums of b_i
    suffix = [0] * (N + 1)
    for i in range(N-1, -1, -1):
        suffix[i] = suffix[i+1] + meds[i][1]
    
    # Binary search for the first day X where sum of b_i for a_i >= X <= K
    low = 1
    high = meds[-1][0] + 1  # Maximum possible day beyond all a_i
    
    while low < high:
        mid = (low + high) // 2
        # Find the first medicine where a_i >= mid
        idx = bisect.bisect_left(meds, (mid, 0))
        total = suffix[idx]
        if total <= K:
            high = mid
        else:
            low = mid + 1
    
    print(low)

if __name__ == '__main__':
    main()