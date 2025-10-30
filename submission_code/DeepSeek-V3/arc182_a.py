MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    P = []
    V = []
    for i in range(Q):
        P.append(int(data[2 + 2*i]))
        V.append(int(data[2 + 2*i + 1]))
    
    # Initialize the sequence S as a list of zeros
    S = [0] * N
    
    # We need to determine the order of operations that allows all operations to be performed without crying
    # The key is to process the operations in a way that ensures no operation tries to set a value lower than the current maximum in the affected range
    
    # To do this, we can process the operations in a specific order, such as by V_i in descending order
    # However, the order of operations is fixed, so we need to find a way to determine the possible choices at each step
    
    # Instead of trying all possible sequences, we can use dynamic programming to count the valid sequences
    
    # Let's consider the operations in the order they are given
    # For each operation, we have two choices: either perform the first type or the second type
    # We need to ensure that the chosen operation does not cause any element in the affected range to be greater than V_i
    
    # To manage this, we can keep track of the current state of the sequence S after each operation
    # However, since N and Q are up to 5000, a direct approach would be too slow
    
    # Instead, we can use the fact that the sequence S is initially all zeros, and each operation sets a range to a specific value
    # We can represent the sequence S as a series of intervals with the same value
    
    # Initialize the sequence as a single interval with value 0
    intervals = [(0, N-1, 0)]
    
    # We will process each operation in order
    # For each operation, we will determine if it can be performed without causing any element in the affected range to be greater than V_i
    # If it can, we will update the intervals accordingly
    
    # We will also keep track of the number of valid sequences
    # Since each operation has two choices, the total number of sequences is 2^Q, but we need to count only those that are valid
    
    # To count the valid sequences, we can use dynamic programming
    # Let dp[i] represent the number of valid sequences after the first i operations
    # Initially, dp[0] = 1 (no operations have been performed yet)
    
    dp = [0] * (Q + 1)
    dp[0] = 1
    
    for i in range(Q):
        p = P[i] - 1  # converting to 0-based index
        v = V[i]
        
        # Determine the maximum value in the affected range for both choices
        # For the first choice: S_1 to S_p
        # For the second choice: S_p to S_N-1
        
        # Find the maximum value in S_1 to S_p
        max_first = 0
        for interval in intervals:
            start, end, val = interval
            if start <= p and end >= 0:
                max_first = max(max_first, val)
        
        # Find the maximum value in S_p to S_N-1
        max_second = 0
        for interval in intervals:
            start, end, val = interval
            if start <= N-1 and end >= p:
                max_second = max(max_second, val)
        
        # Check if the first choice is valid
        if max_first <= v:
            # Update the intervals for the first choice
            # We need to set S_1 to S_p to v
            # This will split the intervals into three parts: before p, p, and after p
            # We need to merge the intervals accordingly
            # For simplicity, we will create a new list of intervals
            new_intervals = []
            for interval in intervals:
                start, end, val = interval
                if end < 0 or start > p:
                    new_intervals.append(interval)
                else:
                    if start < 0:
                        new_intervals.append((start, min(end, -1), val))
                    if start <= p and end >= 0:
                        new_intervals.append((max(start, 0), min(end, p), v))
                    if end > p:
                        new_intervals.append((max(start, p+1), end, val))
            # Remove any intervals with start > end
            new_intervals = [interval for interval in new_intervals if interval[0] <= interval[1]]
            # Update the intervals
            intervals = new_intervals
            # Update the dp
            dp[i+1] += dp[i]
            dp[i+1] %= MOD
        
        # Check if the second choice is valid
        if max_second <= v:
            # Update the intervals for the second choice
            # We need to set S_p to S_N-1 to v
            # This will split the intervals into three parts: before p, p, and after p
            # We need to merge the intervals accordingly
            # For simplicity, we will create a new list of intervals
            new_intervals = []
            for interval in intervals:
                start, end, val = interval
                if end < p or start > N-1:
                    new_intervals.append(interval)
                else:
                    if start < p:
                        new_intervals.append((start, min(end, p-1), val))
                    if start <= N-1 and end >= p:
                        new_intervals.append((max(start, p), min(end, N-1), v))
                    if end > N-1:
                        new_intervals.append((max(start, N), end, val))
            # Remove any intervals with start > end
            new_intervals = [interval for interval in new_intervals if interval[0] <= interval[1]]
            # Update the intervals
            intervals = new_intervals
            # Update the dp
            dp[i+1] += dp[i]
            dp[i+1] %= MOD
    
    print(dp[Q] % MOD)

if __name__ == "__main__":
    main()