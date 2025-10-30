# Read input values
N, L, R = map(int, input().split())

if L > R:
    print("! 0")
    exit()

sum_total = 0

# Calculate sum_0_to_R
current_pos = 0
for i in reversed(range(N + 1)):
    block_size = 1 << i
    if block_size == 0:
        continue
    j = current_pos // block_size
    l = j * block_size
    r_block = l + block_size - 1
    if r_block > R:
        continue
    # Send the query and process the response
    print(f"? {i} {j}")
    t = int(input())
    if t == -1:
        exit()
    sum_total += t
    current_pos = r_block + 1
    if current_pos > R:
        break

# Calculate sum_0_to_L-1 if L > 0
if L > 0:
    x = L - 1
    current_pos = 0
    sum_l = 0
    for i in reversed(range(N + 1)):
        block_size = 1 << i
        if block_size == 0:
            continue
        j = current_pos // block_size
        l = j * block_size
        r_block = l + block_size - 1
        if r_block > x:
            continue
        # Send the query and process the response
        print(f"? {i} {j}")
        t = int(input())
        if t == -1:
            exit()
        sum_l += t
        current_pos = r_block + 1
        if current_pos > x:
            break
    sum_total -= sum_l

# Compute the result and output
result = sum_total % 100
print(f"! {result}")