import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read q and r for each garbage type
garbage_qs = []
garbage_rs = []
for _ in range(N):
    q_val = int(data[index])
    index += 1
    r_val = int(data[index])
    index += 1
    garbage_qs.append(q_val)
    garbage_rs.append(r_val)

# Read Q
Q = int(data[index])
index += 1

# Process each query
for _ in range(Q):
    t = int(data[index])  # t is 1-based
    index += 1
    d = int(data[index])
    index += 1
    
    # Get q and r for the type
    q_t = garbage_qs[t - 1]
    r_t = garbage_rs[t - 1]
    
    # Compute the remainder
    rem = d % q_t
    
    # Compute the difference to the next collection day
    diff = (r_t - rem) % q_t
    
    # Compute the next day
    next_day = d + diff
    
    # Output the result
    print(next_day)