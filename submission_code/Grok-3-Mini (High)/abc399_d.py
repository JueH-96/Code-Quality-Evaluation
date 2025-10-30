import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1

for test_case in range(T):
    N = int(data[index])
    index += 1
    A = list(map(int, data[index:index + 2 * N]))
    index += 2 * N
    
    # Create a list to store positions for each value (1 to N)
    pos_list = [[] for _ in range(N + 1)]
    
    # Assign positions to each value, positions are 1-based
    for pos in range(1, 2 * N + 1):
        val = A[pos - 1]  # A is 0-based, positions are 1-based
        pos_list[val].append(pos)
    
    # Collect non-adjacent values with their min and max positions
    non_adj = []
    for val in range(1, N + 1):
        positions = pos_list[val]
        pos_min = min(positions)
        pos_max = max(positions)
        if pos_max - pos_min > 1:  # Not adjacent
            non_adj.append((pos_min, pos_max))  # Store (min_pos, max_pos)
    
    # Sort the list by min_pos
    non_adj.sort()
    
    # Count the number of valid pairs
    count = 0
    len_non_adj = len(non_adj)
    for i in range(len_non_adj - 1):
        min_curr, max_curr = non_adj[i]
        min_next, max_next = non_adj[i + 1]
        if min_next - min_curr == 1 and abs(max_next - max_curr) == 1:
            count += 1
    
    # Output the count for this test case
    print(count)