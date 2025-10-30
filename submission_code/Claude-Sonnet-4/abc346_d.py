# YOUR CODE HERE
def solve():
    N = int(input())
    S = input().strip()
    C = list(map(int, input().split()))
    
    min_cost = float('inf')
    
    # Try each position as the single matching position
    for target_pos in range(N-1):
        # Try making the matching pair both '0' and both '1'
        for target_char in ['0', '1']:
            cost = 0
            current_s = list(S)
            
            # First, handle the target position
            if current_s[target_pos] != target_char:
                cost += C[target_pos]
                current_s[target_pos] = target_char
            if current_s[target_pos + 1] != target_char:
                cost += C[target_pos + 1]
                current_s[target_pos + 1] = target_char
            
            # Now handle all other adjacent pairs - they must be different
            for i in range(N-1):
                if i == target_pos:
                    continue
                
                if current_s[i] == current_s[i+1]:
                    # Need to flip one of them - choose the cheaper option
                    cost_flip_left = C[i]
                    cost_flip_right = C[i+1]
                    
                    if cost_flip_left <= cost_flip_right:
                        cost += cost_flip_left
                        current_s[i] = '1' if current_s[i] == '0' else '0'
                    else:
                        cost += cost_flip_right
                        current_s[i+1] = '1' if current_s[i+1] == '0' else '0'
            
            min_cost = min(min_cost, cost)
    
    print(min_cost)

solve()