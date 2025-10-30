# YOUR CODE HERE
def solve():
    N, S = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Calculate prefix sums
    prefix = [0]
    for a in A:
        prefix.append(prefix[-1] + a)
    
    total_sum = prefix[N]
    
    # Check all possible contiguous subsequences within one or two periods
    # We need to check subsequences that might span across period boundaries
    
    # First, check subsequences within a single period
    for i in range(N):
        for j in range(i, N):
            subseq_sum = prefix[j+1] - prefix[i]
            if subseq_sum == S:
                print("Yes")
                return
            
            # Check if we can reach S by adding complete cycles
            if total_sum > 0 and S > subseq_sum:
                if (S - subseq_sum) % total_sum == 0:
                    print("Yes")
                    return
    
    # Check subsequences that span across periods
    # These can be represented as: suffix + k*total_sum + prefix
    for i in range(N):
        for j in range(N):
            # Suffix from position i to end of period
            suffix_sum = prefix[N] - prefix[i]
            # Prefix from start to position j
            prefix_sum = prefix[j+1]
            
            # Case 1: suffix + prefix (spanning exactly 2 periods)
            subseq_sum = suffix_sum + prefix_sum
            if subseq_sum == S:
                print("Yes")
                return
            
            # Case 2: suffix + k*total_sum + prefix (k >= 1)
            if total_sum > 0 and S > subseq_sum:
                if (S - subseq_sum) % total_sum == 0:
                    print("Yes")
                    return
    
    print("No")

solve()