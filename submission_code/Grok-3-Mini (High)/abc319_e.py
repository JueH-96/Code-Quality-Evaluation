import sys

# Read all input
data = sys.stdin.read().split()
index = 0

# Read N, X, Y
N = int(data[index])
X = int(data[index + 1])
Y = int(data[index + 2])
index += 3

# Read P and T for each bus
P_list = []
T_list = []
for _ in range(N - 1):
    P = int(data[index])
    T = int(data[index + 1])
    P_list.append(P)
    T_list.append(T)
    index += 2

# Read Q
Q = int(data[index])
index += 1

# Read all queries
queries = []
for _ in range(Q):
    q_i = int(data[index])
    queries.append(q_i)
    index += 1

# Define MOD as LCM of numbers up to 8, which is 840
MOD = 840

# Initialize offset for stop N (arrival time offset to stop N is 0)
offset = [0] * MOD  # This is dp for stop N

# Compute dp backwards from stop N-1 to stop 1
for k in range(N - 1, 0, -1):  # k from N-1 to 1 inclusive
    new_offset = [0] * MOD
    P_bus = P_list[k - 1]  # P for bus from stop k to k+1
    T_bus = T_list[k - 1]  # T for bus from stop k to k+1
    for m in range(MOD):
        r = m % P_bus
        w = 0 if r == 0 else P_bus - r  # Wait time
        delta = w + T_bus  # Time to go from k to k+1
        mod_next = (m + delta) % MOD  # Modulo at next stop
        new_offset[m] = delta + offset[mod_next]  # Offset from k to N
    # Update offset to the new offset for stop k
    offset = new_offset  # Now offset is dp for stop k

# After the loop, offset is dp for stop 1 (offset from stop 1 to stop N)

# Now process each query
for q_i in queries:
    # Arrival time at stop 1
    arr1 = q_i + X
    # Modulo for the arrival time at stop 1
    m = arr1 % MOD
    # Arrival time at stop N
    arrival_N = arr1 + offset[m]
    # Arrival time at Aoki's house
    arrival_aoki = arrival_N + Y
    # Print the result
    print(arrival_aoki)