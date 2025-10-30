import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    S = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    # Check all possible contiguous subarrays within the first period
    prefix = [0]
    current_sum = 0
    seen = set()
    seen.add(0)
    found = False
    for num in A:
        current_sum += num
        if (current_sum - S) in seen:
            print("Yes")
            return
        seen.add(current_sum)
        prefix.append(current_sum)
    
    # If not found, proceed to check wrap-around cases
    total_sum = prefix[-1]
    if total_sum == 0:
        # All elements are zero, but since all A_i >=1, this is impossible
        print("No")
        return
    
    # Create a dictionary to map remainders to list of prefix sums
    remainder_map = {}
    for p in prefix:
        r = p % total_sum
        if r not in remainder_map:
            remainder_map[r] = []
        remainder_map[r].append(p)
    
    # Sort each list for binary search
    for r in remainder_map:
        remainder_map[r].sort()
    
    for i in range(N):
        sum_i = prefix[-1] - prefix[i]
        target = S - sum_i
        if target < 0:
            continue
        r = target % total_sum
        if r not in remainder_map:
            continue
        lst = remainder_map[r]
        # Find the largest p <= target
        idx_bisect = bisect.bisect_right(lst, target) - 1
        if idx_bisect >= 0:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()