# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # If A is already all 1s, any string is good
    if all(a == 1 for a in A):
        return "Yes"
    
    # Create a mapping from position to position mod 3
    position_to_mod = {}
    
    # For each zero in A, check if it can be covered
    for i in range(N):
        if A[i] == 0:
            # 1-indexed positions for the problem
            pos_1_indexed = i + 1
            
            # Determine the mod 3 class of this position
            mod_3 = pos_1_indexed % 3
            position_to_mod[pos_1_indexed] = mod_3
    
    # Check if zeros are in positions that form a complete congruence class mod 3
    mod_0_exists = any(mod == 0 for mod in position_to_mod.values())
    mod_1_exists = any(mod == 1 for mod in position_to_mod.values())
    mod_2_exists = any(mod == 2 for mod in position_to_mod.values())
    
    # If zeros exist in all three congruence classes mod 3, no good string exists
    if mod_0_exists and mod_1_exists and mod_2_exists:
        return "No"
    
    # Otherwise, a good string exists
    return "Yes"

print(solve())