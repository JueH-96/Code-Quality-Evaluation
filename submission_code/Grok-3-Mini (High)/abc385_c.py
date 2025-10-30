import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
H = list(map(int, data[index:index + N]))
index += N

# Create height to positions mapping
height_to_positions = [[] for _ in range(3001)]
for pos in range(1, N + 1):
    h = H[pos - 1]
    height_to_positions[h].append(pos)  # positions are added in order, so sorted

# Initialize answer
ans = 1

# Iterate over all possible heights
for h in range(1, 3001):
    group = height_to_positions[h]
    if not group:
        continue
    pos_list = group
    pos_set = set(pos_list)  # for fast lookup
    K = len(pos_list)
    
    # Compute max AP length for this group
    max_ap_len = 1
    
    # For each possible difference D
    for D in range(1, N):  # D from 1 to N-1
        # For each position, check if it has no predecessor
        for pos in pos_list:
            if (pos - D) not in pos_set:  # no building at pos - D
                # Start a chain from pos with difference D
                len_chain = 1  # include pos itself
                current_val = pos + D
                while current_val in pos_set:
                    len_chain += 1
                    current_val += D  # move to the next potential position
                # Update max_ap_len for this group
                max_ap_len = max(max_ap_len, len_chain)
    
    # After checking all D, update the global answer
    ans = max(ans, max_ap_len)

# After processing all groups, output the answer
print(ans)