def min_operations_to_rearrange(N, S, T):
    # Count the number of B's and W's in S and T
    count_S_B = S.count('B')
    count_T_B = T.count('B')
    count_S_W = S.count('W')
    count_T_W = T.count('W')

    # If the counts of B's and W's do not match, it's impossible
    if count_S_B != count_T_B or count_S_W != count_T_W:
        return -1

    # We need to match the positions of B's and W's from S to T
    # We will use a greedy approach to find the minimum number of swaps
    # We will consider the problem as matching pairs of stones to be moved to the empty cells

    # Convert S and T into lists for easier manipulation
    S = list(S)
    T = list(T)

    # We will count the number of swaps needed
    swaps = 0

    # We will use a two-pointer approach to find mismatches and try to resolve them
    i = 0
    while i < N:
        if S[i] != T[i]:
            # Find the next position where S[j] == T[i] and S[i] != T[j]
            j = i + 1
            while j < N and (S[j] == S[i] or T[j] != T[i]):
                j += 1
            if j < N:
                # Swap S[i] with S[j]
                S[i], S[j] = S[j], S[i]
                swaps += 1
        i += 1

    return swaps

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
T = data[2]

result = min_operations_to_rearrange(N, S, T)
print(result)