def solve():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    medicines = []
    idx = 2
    for _ in range(N):
        a_i = int(input_data[idx]); b_i = int(input_data[idx+1])
        idx += 2
        medicines.append((a_i, b_i))
    
    # Sort by a_i
    medicines.sort(key=lambda x: x[0])
    
    # Prepare arrays for prefix sums
    A = [m[0] for m in medicines]
    B = [m[1] for m in medicines]
    
    # Prefix sums of B
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + B[i]
    total_sum = prefix[N]
    
    # day_sum(x) = sum of b_i for all i s.t. a_i >= x
    # We can find idx = bisect_left(A, x), so those from idx to end are >= x
    # sum = total_sum - prefix[idx]
    def day_sum(x):
        pos = bisect.bisect_left(A, x)
        return total_sum - prefix[pos]
    
    # Binary search in [1, max(A) + 2]
    # We want the smallest day >= 1 such that day_sum(day) <= K
    left = 1
    right = max(A) + 2  # an upper bound where day_sum is definitely 0
    while left < right:
        mid = (left + right) // 2
        if day_sum(mid) <= K:
            right = mid
        else:
            left = mid + 1
    
    # left will be the first day (>=1) with day_sum(left) <= K
    print(left)
    
def main():
    solve()

if __name__ == "__main__":
    main()