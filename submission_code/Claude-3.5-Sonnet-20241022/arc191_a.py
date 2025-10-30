def solve():
    # Read input
    N, M = map(int, input().split())
    S = list(input().strip())
    T = input().strip()
    
    # For each position in T, we want to maximize the resulting number
    # We'll try to place larger digits in more significant positions
    for k in range(M):
        # Current digit we need to place
        curr_digit = T[k]
        
        # Find the best position to place this digit
        best_pos = -1
        max_result = S[:]  # Copy current state
        
        # Try placing the digit in each position
        for i in range(N):
            # Make a copy of current state
            temp = S[:]
            temp[i] = curr_digit
            
            # If this gives us a better result, update best position
            if temp > max_result:
                max_result = temp
                best_pos = i
        
        # If we found a position that improves the number, make the change
        if best_pos != -1:
            S[best_pos] = curr_digit
    
    # Convert final list back to string and return
    return ''.join(S)

# Main execution
if __name__ == "__main__":
    result = solve()
    print(result)