# YOUR CODE HERE
S = input().strip()
N = int(input().strip())

# If no '?' in S, just check the value
if '?' not in S:
    val = int(S, 2)
    if val <= N:
        print(val)
    else:
        print(-1)
else:
    # Try to find the largest valid number
    result = -1
    
    # Convert S to a list for easier manipulation
    s_list = list(S)
    
    def solve(pos, current_list):
        global result
        
        # If we've processed all positions
        if pos == len(s_list):
            val = int(''.join(current_list), 2)
            if val <= N:
                result = max(result, val)
            return
        
        # If current position is not '?'
        if s_list[pos] != '?':
            current_list[pos] = s_list[pos]
            # Check if we can continue (early pruning)
            # Build the maximum possible value from current position
            temp = current_list[:]
            for i in range(pos + 1, len(s_list)):
                if s_list[i] == '?':
                    temp[i] = '1'
                else:
                    temp[i] = s_list[i]
            max_possible = int(''.join(temp), 2)
            
            if max_possible <= N:
                solve(pos + 1, current_list)
        else:
            # Try '1' first (to get larger values)
            current_list[pos] = '1'
            # Check if we can continue
            temp = current_list[:]
            for i in range(pos + 1, len(s_list)):
                if s_list[i] == '?':
                    temp[i] = '0'  # Use minimum for checking
                else:
                    temp[i] = s_list[i]
            min_possible = int(''.join(temp), 2)
            
            if min_possible <= N:
                solve(pos + 1, current_list)
            
            # Try '0'
            current_list[pos] = '0'
            solve(pos + 1, current_list)
    
    # Start solving
    result = -1
    solve(0, ['0'] * len(s_list))
    print(result)