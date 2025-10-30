# Read the input integer B
B = int(input())

ans = -1

# We are looking for a positive integer A such that A^A = B.
# The function A^A is strictly increasing for A >= 1.
# A=1: 1^1 = 1
# ...
# A=15: 15^15 = 437,893,890,380,859,375 (approx 4.37 * 10^17)
# A=16: 16^16 = 18,446,744,073,709,551,616 (approx 1.84 * 10^19)
# Since B <= 10^18, A cannot be 16 or greater.
# So, we only need to check A from 1 up to 15.
# The loop range `range(1, 17)` means A will take values 1, 2, ..., 16.
# If A reaches 16, A^A (16^16) will be > B (since B_max is 10^18),
# and the `if val > B:` condition will break the loop.
for A in range(1, 17):
    # Calculate A^A. Python's ** operator handles large integers.
    val = A**A
    
    if val == B:
        ans = A
        break  # Found the unique solution
    
    if val > B:
        # If A^A exceeds B, no larger A can be a solution
        # because A^A is strictly increasing.
        break
        
# Print the result
print(ans)