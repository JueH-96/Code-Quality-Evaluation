import sys

# Read N, L, R from standard input
line = sys.stdin.readline().strip()
N, L, R = map(int, line.split())

# Initialize current left position and total sum
current_left = L
total_sum = 0

# Loop until the entire range [L, R] is covered
while current_left <= R:
    # Compute the current length of the remaining range
    len_current = R - current_left + 1
    
    # Compute the maximum i for the dyadic interval
    if current_left == 0:
        max_i = (len_current).bit_length() - 1
    else:
        low_bit = current_left & (-current_left)
        tz = (low_bit.bit_length() - 1)  # Number of trailing zeros
        floor_log_len = (len_current).bit_length() - 1
        max_i = min(tz, floor_log_len)
    
    # Set i and compute j
    i = max_i
    j = current_left >> i  # Right shift to get j
    
    # Ask the query
    print(f"? {i} {j}", flush=True)
    
    # Read the response
    response_line = sys.stdin.readline().strip()
    T = int(response_line)
    
    # If invalid response, terminate
    if T == -1:
        sys.exit(0)
    
    # Add the sum modulo 100
    total_sum += T
    total_sum %= 100
    
    # Move to the next position
    size = 1 << i
    current_left += size

# Output the answer
print(f"! {total_sum}", flush=True)