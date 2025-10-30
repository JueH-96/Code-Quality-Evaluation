def main():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Adjust for 1-indexing
    A = [0] + A
    memo = {}
    
    # Function to compute the minimum number of operations for a subarray
    def min_operations(L, R):
        # Create a frequency map of the subarray
        freq = {}
        for i in range(L, R+1):
            freq[A[i]] = freq.get(A[i], 0) + 1
        
        def solve(freq_dict):
            if not freq_dict:
                return 0
            
            # Convert the dictionary to a tuple of (key, value) pairs for hashing
            freq_tuple = tuple(sorted(freq_dict.items()))
            if freq_tuple in memo:
                return memo[freq_tuple]
            
            min_ops = float('inf')
            keys = sorted(freq_dict.keys())
            
            # Try all possible starting points and ending points
            for i, l in enumerate(keys):
                for j in range(i, len(keys)):
                    r = keys[j]
                    # Check if all integers from l to r are on the blackboard
                    if j - i + 1 == r - l + 1:
                        # Create a new frequency map after erasing
                        new_freq = {k: v for k, v in freq_dict.items() if k < l or k > r}
                        min_ops = min(min_ops, 1 + solve(new_freq))
            
            memo[freq_tuple] = min_ops
            return min_ops
        
        return solve(freq)
    
    total = 0
    for L in range(1, N+1):
        for R in range(L, N+1):
            total += min_operations(L, R)
    print(total)

if __name__ == "__main__":
    main()