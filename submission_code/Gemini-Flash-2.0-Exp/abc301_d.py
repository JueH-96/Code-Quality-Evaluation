def solve():
    s = input()
    n = int(input())
    
    q_indices = [i for i, char in enumerate(s) if char == '?']
    num_q = len(q_indices)
    
    max_val = -1
    
    for i in range(2**num_q):
        temp_s = list(s)
        binary_rep = bin(i)[2:].zfill(num_q)
        
        for j in range(num_q):
            temp_s[q_indices[j]] = binary_rep[j]
        
        curr_val = int("".join(temp_s), 2)
        
        if curr_val <= n:
            max_val = max(max_val, curr_val)
            
    print(max_val)

solve()