# YOUR CODE HERE
def can_transform(K, S, T):
    len_S = len(S)
    len_T = len(T)
    # Calculate the length difference
    len_diff = abs(len_S - len_T)
    # If the length difference is more than K, it's impossible
    if len_diff > K:
        return False
    # Calculate the number of mismatches
    mismatches = 0
    min_len = min(len_S, len_T)
    for i in range(min_len):
        if S[i] != T[i]:
            mismatches += 1
    # Total operations needed is mismatches plus the length difference
    total_ops = mismatches + len_diff
    return total_ops <= K

# Read input
K = int(input())
S = input().strip()
T = input().strip()

# Determine if transformation is possible
if can_transform(K, S, T):
    print("Yes")
else:
    print("No")