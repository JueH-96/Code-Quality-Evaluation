import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N
N = int(data[index])
index += 1

# Read the array A
A = list(map(int, data[index:index + N]))

# Initialize answer with the number of single element subarrays
ans = N

# If N > 1, compute the differences and find runs of equal differences
if N > 1:
    # Initialize the first difference and run length
    prev_diff_val = A[1] - A[0]
    current_run_len = 1
    
    # Iterate through the difference indices starting from 1
    for i in range(1, N - 1):  # i ranges from 1 to N-2
        current_diff_val = A[i + 1] - A[i]
        if current_diff_val == prev_diff_val:
            current_run_len += 1
        else:
            # Add the number of subarrays for the ended run
            ans += (current_run_len * (current_run_len + 1)) // 2
            # Reset for the new run
            prev_diff_val = current_diff_val
            current_run_len = 1
    
    # Add the number of subarrays for the last run
    ans += (current_run_len * (current_run_len + 1)) // 2

# Output the answer
print(ans)