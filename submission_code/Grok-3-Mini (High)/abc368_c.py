import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
H = [int(data[i]) for i in range(index, index + N)]

# Helper function to find minimum number of attacks M
def min_attacks(R, health):
    left = 1
    right = health
    while left <= right:
        mid = (left + right) // 2
        Q_mid = mid // 3
        Rm_mid = mid % 3
        if R < Rm_mid:
            num_R_mid = Q_mid + 1
        else:
            num_R_mid = Q_mid
        C_mid = mid + 2 * num_R_mid
        if C_mid >= health:
            right = mid - 1
        else:
            left = mid + 1
    return left  # Returns the smallest M such that cumulative damage >= health

# Initialize starting T for the first enemy
current_T_start = 1

# Iterate through each enemy's health
for h in H:
    # Compute the phase S and R for the current T_start
    S = current_T_start % 3
    R = (-S) % 3
    # Find the number of attacks M needed for this enemy
    M = min_attacks(R, h)
    # Update T_start for the next enemy
    current_T_start += M

# The final T is one less than the current T_start after all updates
answer = current_T_start - 1

# Output the answer
print(answer)