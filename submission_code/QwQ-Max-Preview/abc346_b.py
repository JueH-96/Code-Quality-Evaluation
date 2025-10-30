W, B = map(int, input().split())
L = W + B

# Precompute the prefix sums for the base pattern "wbwbwwbwbwbw"
prefix_w = [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7]
prefix_b = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5]
total_w_base = 7
total_b_base = 5

found = False

for s in range(12):
    initial_length = min(L, 12 - s)
    initial_start = s
    initial_end = s + initial_length
    
    # Calculate initial part's w and b counts
    iw = prefix_w[initial_end] - prefix_w[initial_start]
    ib = prefix_b[initial_end] - prefix_b[initial_start]
    
    remaining_L = L - initial_length
    if remaining_L > 0:
        full_cycles = remaining_L // 12
        remaining_after_full = remaining_L % 12
        fw = prefix_w[remaining_after_full]
        fb = prefix_b[remaining_after_full]
    else:
        full_cycles = 0
        fw = 0
        fb = 0
    
    total_w = iw + full_cycles * total_w_base + fw
    total_b = ib + full_cycles * total_b_base + fb
    
    if total_w == W and total_b == B:
        found = True
        break

print("Yes" if found else "No")