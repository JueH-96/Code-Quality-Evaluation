import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1
S_str = data[index]
index += 1
S_list = list(S_str)

# Compute initial count of "ABC"
count = 0
for i in range(N - 2):
    if S_list[i] == 'A' and S_list[i + 1] == 'B' and S_list[i + 2] == 'C':
        count += 1

# Process each query
for _ in range(Q):
    X = int(data[index])  # 1-based index
    index += 1
    C_new = data[index]   # new character
    index += 1
    idx_pos = X - 1  # 0-based index
    
    # Find the range of affected start indices i
    low = max(0, idx_pos - 2)
    high = min(idx_pos, N - 3)
    net_change = 0
    
    for i in range(low, high + 1):
        # Check old triplet
        old_is_abc = (S_list[i] == 'A' and S_list[i + 1] == 'B' and S_list[i + 2] == 'C')
        
        # Compute new triplet with change
        new_A = C_new if i == idx_pos else S_list[i]
        new_B = C_new if i + 1 == idx_pos else S_list[i + 1]
        new_C = C_new if i + 2 == idx_pos else S_list[i + 2]
        new_is_abc = (new_A == 'A' and new_B == 'B' and new_C == 'C')
        
        if old_is_abc and not new_is_abc:
            net_change -= 1
        elif not old_is_abc and new_is_abc:
            net_change += 1
    
    # Update count
    count += net_change
    
    # Apply the change to the string
    S_list[idx_pos] = C_new
    
    # Output the current count
    print(count)