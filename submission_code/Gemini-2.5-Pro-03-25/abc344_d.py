# YOUR CODE HERE
import sys

def solve():
    # Read target string T from standard input
    T = sys.stdin.readline().strip()
    # Read number of bags N from standard input
    N = int(sys.stdin.readline())
    
    bags = []
    # Read the contents of each bag
    for _ in range(N):
        # Read the line containing A_i and strings S_{i,j}
        line = sys.stdin.readline().split()
        # The first element line[0] is A_i, the count of strings. We don't need it explicitly 
        # as we can just use the length of the list of strings.
        # The rest of the elements line[1:] are the strings S_{i,j} for bag i.
        S_i = line[1:]
        bags.append(S_i)

    # Length of the target string T
    L = len(T)
    
    # Initialize DP array. dp[j] will store the minimum cost to form the prefix T[0...j-1].
    # The size of the DP array needs to be L+1 to accommodate lengths from 0 to L.
    
    # We need a value to represent infinity. Since the maximum possible cost is N (if we pick one 
    # string from each bag), any value greater than N will work. N <= 100.
    # Let's use INF = N + 1 = 101 as a safe value for infinity.
    INF = N + 1 
    
    dp = [INF] * (L + 1)
    
    # Base case: The empty prefix (length 0) can be formed with cost 0 before considering any bags.
    dp[0] = 0
    
    # Iterate through each bag from index 0 to N-1
    for i in range(N):
        current_bag = bags[i]
        
        # We iterate through possible current prefix lengths k from L down to 0.
        # Iterating downwards allows for safe in-place updates of the dp array.
        # The logic is that when we compute an update for dp[k + L_m] based on dp[k],
        # the value dp[k] represents the minimum cost *before* considering the options from bag i 
        # starting from state k. Updates affect states with index greater than k. 
        # Since we iterate k downwards, the computation for dp[k'] where k' < k will use the 
        # state values from before processing bag i, which is correct.
        for k in range(L, -1, -1):
            
            # If the state corresponding to prefix length k is unreachable (cost is INF), 
            # we cannot extend from this state, so we skip it.
            if dp[k] == INF:
                continue
            
            # Option 1: Do nothing.
            # This action doesn't change the prefix length `k` or the cost `dp[k]`. 
            # In the in-place update approach, this case is implicitly handled because `dp[k]` 
            # retains its value from the previous iteration unless it's updated by Option 2 below.

            # Option 2: Choose a string S_m from the current bag i and concatenate it.
            for S_m in current_bag:
                L_m = len(S_m)
                
                # Check if appending S_m would exceed the length of the target string T.
                if k + L_m <= L:
                    # Check if the substring of T starting at index k matches the chosen string S_m.
                    # Python slicing T[k : k + L_m] extracts the substring of length L_m starting at index k.
                    if T[k : k + L_m] == S_m:
                        # If it matches, we have found a way to reach the prefix of length k + L_m.
                        # The cost to reach this new state is the cost to reach state k plus 1 (for picking S_m).
                        # We update dp[k + L_m] with the minimum cost found so far.
                        dp[k + L_m] = min(dp[k + L_m], dp[k] + 1)

    # After iterating through all N bags, dp[L] contains the minimum cost to form the full string T.
    final_cost = dp[L]
    
    # If dp[L] is still INF, it means T could not be formed following the rules.
    if final_cost == INF:
        print("-1")
    else:
        # Otherwise, print the minimum cost found.
        print(final_cost)

# Execute the solve function when the script is run
if __name__ == '__main__':
    solve()