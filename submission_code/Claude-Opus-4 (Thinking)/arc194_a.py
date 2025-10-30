n = int(input())
a = list(map(int, input().split()))

# current: dictionary mapping S (as tuple) to its sum
current = {tuple(): 0}

for i in range(n):
    next_states = {}
    
    for s_tuple, s_sum in current.items():
        # Option 1: Append A[i]
        new_s = s_tuple + (a[i],)
        new_sum = s_sum + a[i]
        if new_s not in next_states or next_states[new_s] < new_sum:
            next_states[new_s] = new_sum
        
        # Option 2: Delete last element (if S is not empty)
        if len(s_tuple) > 0:
            new_s = s_tuple[:-1]
            new_sum = s_sum - s_tuple[-1]
            if new_s not in next_states or next_states[new_s] < new_sum:
                next_states[new_s] = new_sum
    
    current = next_states

# Find maximum sum among all possible final states
print(max(current.values()))