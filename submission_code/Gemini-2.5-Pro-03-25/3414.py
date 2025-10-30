import math

class Solution:
    """
    Calculates the number of ways Alice can reach stair k starting from stair 1.
    Alice starts at stair 1 with jump = 0.
    Operations:
    1. Go down 1 stair (i -> i-1): Cannot be used consecutively or on stair 0.
    2. Go up 2^jump stairs (i -> i + 2^jump): After this operation, jump becomes jump + 1.

    Approach:
    Let N be the total number of Up operations performed in a sequence of moves.
    The sequence of jump values used for these Up operations will be 0, 1, 2, ..., N-1.
    The total displacement from N Up operations is the sum of 2^j for j from 0 to N-1, which equals 2^N - 1.
    
    Let M be the total number of Down operations performed in the sequence.
    Each Down operation decreases the stair number by 1. The total displacement from M Down operations is -M.

    Alice starts at stair 1. After N Up operations and M Down operations, her final stair position will be:
    Final Stair = Initial Stair + Total Displacement from Ups + Total Displacement from Downs
    Final Stair = 1 + (2^N - 1) + (-M) = 2^N - M.

    We want the final stair to be k. Therefore, we must have the equation:
    2^N - M = k
    This equation gives the required number of Down operations M for a given number of Up operations N:
    M = 2^N - k

    Now we consider the constraints on the operations:
    1.  The number of Down operations M must be non-negative. This implies M >= 0, which means 2^N - k >= 0, or 2^N >= k. This gives a lower bound on N. Any N for which 2^N < k is invalid.
    
    2.  The Down operation cannot be used consecutively. This means that between any two Down operations, there must be at least one Up operation. Also, the first operation can be Down. This constraint implies that the total number of Down operations M cannot exceed the number of Up operations N plus one. Why? Consider N Up operations arranged like U U ... U. There are N+1 possible slots (before the first U, between any two Us, and after the last U) where Down operations can be inserted without any two Downs being adjacent: _ U _ U _ ... _ U _. To place M Down operations into these slots, we must have M <= N+1.

    Combining these constraints, for a given N, the required M = 2^N - k must satisfy 0 <= M <= N+1.

    If these conditions are met for a pair (N, M), we need to find the number of distinct sequences of N Up operations and M Down operations that satisfy the non-consecutive Down constraint. This is a combinatorial problem: choose M slots out of N+1 available slots to place the M Down operations. The number of ways is given by the binomial coefficient "N+1 choose M", denoted as C(N+1, M).

    The total number of ways to reach stair k is the sum of C(N+1, M) over all possible values of N for which the derived M satisfies the constraints 0 <= M <= N+1.

    Algorithm:
    Iterate through possible values of N starting from 0.
    For each N, calculate the required M = 2^N - k.
    Check if M is valid:
    - If M < 0, then 2^N < k. This N is too small. Continue to the next N.
    - If M > N + 1, then 2^N - k > N + 1. Since the function f(N) = 2^N - N - 1 grows rapidly for N >= 1, this condition will also hold for all larger N. So, we can stop iterating.
    - If 0 <= M <= N+1, then this pair (N, M) is valid. Calculate C(N+1, M) using `math.comb(N + 1, M)` and add it to the total count.

    The loop for N needs an upper bound. Since k <= 10^9, N needs to be at least ceil(log2(k)), which is around 30. The condition M <= N+1 (or 2^N - k <= N+1) will typically only hold for N slightly larger than log2(k). Based on analysis, N around 30-31 is the maximum relevant value for k=10^9. A loop limit like 35 is safe and sufficient.
    """
    def waysToReachStair(self, k: int) -> int:
        # Initialize total ways count
        ans = 0
        
        # Iterate through possible number of Up operations N.
        # Using a range up to 35, which is sufficient for k <= 10^9 based on analysis.
        # The loop will likely break much earlier due to the M > N+1 condition.
        for N in range(35): 
            # Calculate 2^N using bit shift for efficiency
            power_of_2_N = 1 << N
            
            # Calculate the required number of Down operations M to reach stair k
            M = power_of_2_N - k
            
            # Validate M based on derived constraints:
            
            # Constraint 1: M must be non-negative. (Equiv. 2^N >= k)
            if M < 0:
                # If M < 0, this N is too small. Continue to check the next larger N.
                continue 
                
            # Constraint 2: M must not exceed N + 1. (Due to non-consecutive Down rule)
            if M > N + 1:
                # If M > N + 1, this N (and all larger N's) are invalid because
                # they require too many Down operations relative to Up operations.
                # Break the loop as no further N will satisfy the constraints.
                break 

            # If M is valid (i.e., 0 <= M <= N + 1), then this pair (N, M) contributes to the total ways.
            # Calculate the number of ways using combinations: C(N + 1, M).
            # This counts the number of valid sequences of N Ups and M Downs.
            # math.comb(n, k) calculates "n choose k". It requires 0 <= k <= n.
            # Our checks ensure 0 <= M <= N + 1, so the arguments are valid.
            ways = math.comb(N + 1, M)
            
            # Add the number of ways for this valid N to the total count.
            ans += ways

        # Return the total accumulated ways.
        return ans