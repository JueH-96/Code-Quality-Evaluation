import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to parse the data
index = 0

# Read N, M, K
N = int(data[index])
index += 1
M = int(data[index])
index += 1
K = int(data[index])
index += 1

# Read the M tests and store the mask and result for each test
tests = []
for _ in range(M):
    C_i = int(data[index])
    index += 1
    mask_i = 0
    for _ in range(C_i):
        key = int(data[index])
        index += 1
        # Set the bit corresponding to the key (key is 1-based, so use key-1)
        mask_i |= (1 << (key - 1))
    R_i = data[index]  # Read the result as string 'o' or 'x'
    index += 1
    tests.append((mask_i, R_i))

# Now iterate over all possible subsets S (using bitmasks)
count = 0
for S in range(1 << N):  # 2^N subsets
    valid = True
    for mask_i, R_i in tests:
        # Compute the intersection size
        inter_mask = S & mask_i
        inter_count = bin(inter_mask).count('1')  # Count the number of set bits
        # Check the condition based on R_i
        if (R_i == 'o' and inter_count < K) or (R_i == 'x' and inter_count >= K):
            valid = False
            break  # No need to check further tests for this S
    if valid:
        count += 1

# Output the count
print(count)