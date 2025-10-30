import bisect

def main():
    import sys
    N, S = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Case 1: Check for any subarray in A with sum S using sliding window
    current_sum = 0
    left = 0
    for right in range(len(A)):
        current_sum += A[right]
        while current_sum > S and left <= right:
            current_sum -= A[left]
            left += 1
        if current_sum == S:
            print("Yes")
            return
    
    # Precompute prefix sums
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = prefix[i] + A[i]
    total_sum = prefix[-1]
    
    # Compute right_sum and left_sum
    right_sum = [prefix[N] - prefix[i] for i in range(N)]
    left_sum = [prefix[i+1] for i in range(N)]
    
    # Case 2: Handle total_sum == 0
    if total_sum == 0:
        left_set = set(left_sum)
        for i in range(N):
            target = S - right_sum[i]
            if target in left_set:
                print("Yes")
                return
    else:
        # Build mod_dict with sorted lists
        mod_dict = {}
        for j in range(N):
            l_sum = left_sum[j]
            rem = l_sum % total_sum
            if rem not in mod_dict:
                mod_dict[rem] = []
            mod_dict[rem].append(l_sum)
        # Sort each list in mod_dict
        for rem in mod_dict:
            mod_dict[rem].sort()
        # Check each i
        for i in range(N):
            target = S - right_sum[i]
            rem = (target % total_sum + total_sum) % total_sum
            if rem not in mod_dict:
                continue
            lst = mod_dict[rem]
            # Binary search for the largest element <= target
            idx = bisect.bisect_right(lst, target) - 1
            if idx >= 0:
                print("Yes")
                return
    
    # If all checks fail
    print("No")

if __name__ == "__main__":
    main()