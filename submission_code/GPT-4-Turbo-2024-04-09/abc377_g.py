import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    strings = data[1:]
    
    results = []
    
    # Use a set to store previous strings for quick lookup
    previous_strings = set()
    
    for i in range(N):
        current_string = strings[i]
        current_length = len(current_string)
        
        # Calculate cost to make current string empty
        min_cost = current_length
        
        # Check cost to transform current string to any of the previous strings
        for prev_string in previous_strings:
            # Calculate the cost to transform current_string to prev_string
            # This is the edit distance where only insertions and deletions are allowed
            # and both have a cost of 1.
            len_prev = len(prev_string)
            # Use dynamic programming to find the minimum edit distance
            dp = [[0] * (len_prev + 1) for _ in range(2)]
            for j in range(len_prev + 1):
                dp[0][j] = j
            
            for j in range(1, current_length + 1):
                dp[j % 2][0] = j
                for k in range(1, len_prev + 1):
                    if current_string[j - 1] == prev_string[k - 1]:
                        dp[j % 2][k] = dp[(j - 1) % 2][k - 1]
                    else:
                        dp[j % 2][k] = min(dp[(j - 1) % 2][k] + 1, dp[j % 2][k - 1] + 1)
            
            edit_distance = dp[current_length % 2][len_prev]
            min_cost = min(min_cost, edit_distance)
        
        # Add the current string to the set of previous strings
        previous_strings.add(current_string)
        
        # Append the result for the current string
        results.append(min_cost)
    
    # Print all results
    for result in results:
        print(result)