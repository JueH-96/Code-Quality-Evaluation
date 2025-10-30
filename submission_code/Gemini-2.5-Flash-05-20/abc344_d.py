import sys

def solve():
    # Read the target string T
    T = sys.stdin.readline().strip()
    # Read the number of bags N
    N = int(sys.stdin.readline())

    L = len(T)

    # dp[j] will store the minimum cost to form the prefix T[:j].
    # Initialize all costs to infinity (float('inf')), as they are initially unachievable.
    # dp[0] is 0, representing the cost to form an empty string.
    dp = [float('inf')] * (L + 1)
    dp[0] = 0

    # Iterate through each of the N bags
    for _ in range(N):
        line = sys.stdin.readline().split()
        # A_i is the count of strings in the current bag (line[0] converted to int)
        # strings_in_bag contains all S_{i,j} for the current bag (remaining elements in line)
        strings_in_bag = line[1:]

        # Create a temporary dp array for updates based on the current bag.
        # Initialize next_dp with a copy of the current dp values.
        # This implicitly covers the "do nothing" option for the current bag,
        # meaning all current achievable costs are carried over without change.
        next_dp = list(dp) 

        # Iterate through each string s_k available in the current bag
        for s_k in strings_in_bag:
            len_k = len(s_k)
            
            # Iterate through all possible starting positions 'j' in T.
            # We are checking if s_k can be appended to a prefix T[:j] that was already formed.
            # 'j' can range from 0 up to L - len_k.
            for j in range(L - len_k + 1):
                # Check if the prefix T[:j] was formable (i.e., its cost is not infinity).
                if dp[j] != float('inf'):
                    # Check if the string s_k matches the corresponding substring of T,
                    # starting at index j and having length len_k.
                    if T[j : j + len_k] == s_k:
                        # If it matches, we can potentially form T[:j + len_k].
                        # The new cost would be dp[j] (cost for T[:j]) + 1 (for picking s_k).
                        # We update next_dp[j + len_k] with the minimum of its current value
                        # and this new potential cost.
                        next_dp[j + len_k] = min(next_dp[j + len_k], dp[j] + 1)
        
        # After considering all strings in the current bag and the "do nothing" option,
        # update the main dp array with the calculated next_dp values.
        # This prepared dp for the next bag's iteration.
        dp = next_dp

    # The final answer is the minimum cost to form the entire string T, which is dp[L].
    result = dp[L]

    # If dp[L] is still infinity, it means T cannot be formed using the given bags and rules.
    if result == float('inf'):
        print(-1)
    else:
        print(result)

# Call the solve function to execute the program
solve()