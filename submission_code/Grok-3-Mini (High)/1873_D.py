import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0
t = int(data[index])
index += 1

# Process each test case
for _ in range(t):
    n = int(data[index])
    index += 1
    k = int(data[index])
    index += 1
    s = data[index]
    index += 1
    
    # Find positions of black cells
    pos = [i for i in range(n) if s[i] == 'B']
    m = len(pos)
    
    # Initialize answer and index
    ans = 0
    idx = 0
    
    # Greedy algorithm to find minimum operations
    while idx < m:
        ans += 1
        current_pos = pos[idx]
        right_end = min(n - 1, current_pos + k - 1)
        # Move index to the first position greater than right_end
        while idx < m and pos[idx] <= right_end:
            idx += 1
    
    # Output the answer for this test case
    print(ans)