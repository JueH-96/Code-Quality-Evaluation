import sys

# Read N, M
line1 = sys.stdin.readline().split()
N = int(line1[0])
M = int(line1[1])

# Read S and T
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

s_list = list(S)

# Group indices of T by digit, sorted descending by index.
# Iterating through T from right to left guarantees that the first index
# we encounter for a digit in the list t_indices_by_digit[d_char]
# is the latest occurrence of that digit in T.
t_indices_by_digit = {d: [] for d in '123456789'}
for k in range(M - 1, -1, -1):
    t_indices_by_digit[T[k]].append(k)

# Boolean array to track which operations from T have been used as the final update
# for a position processed so far (from left to right).
# An operation k (0-indexed) at T[k], performed at step k+1, can only be the final update for position j
# if we choose index j+1 for operation k+1 AND no operation k' > k is also chosen for index j+1.
# By processing positions from j=0 to N-1 and greedily picking the latest
# available operation from the remaining set for the current position j,
# we ensure that the chosen operation k will indeed be the last one for position j
# among the operations that were *not* used as the last update for positions 0..j-1.
# Any operation used for position j cannot be the last update for any position j' > j.
used_ops = [False] * M

# Iterate through positions in S from left to right
for j in range(N):
    current_s_digit_char = s_list[j]
    current_s_digit_int = int(current_s_digit_char)

    best_digit_char = current_s_digit_char
    best_op_idx = -1 # -1 means original digit is kept

    # Iterate through possible digits for replacement from 9 down to current_s_digit_int + 1
    # We are looking for the largest digit from T that can improve the current s_list[j].
    for d_int in range(9, current_s_digit_int, -1):
        d_char = str(d_int)
        
        # Check if there are any operations in T with this digit that haven't been used
        # as the final update for a previous position.
        if d_char in t_indices_by_digit:
            # Find the latest available operation index for this digit.
            # t_indices_by_digit[d_char] stores indices sorted descending.
            # The first index in the list is the latest occurrence in T.
            while t_indices_by_digit[d_char]:
                op_idx = t_indices_by_digit[d_char][0]
                
                if not used_ops[op_idx]:
                    # This operation (at original index op_idx in T) is available.
                    # It provides a digit d_char > current_s_digit_int.
                    # Since we iterate d_int downwards from 9, the first d_int for which we find an available op provides the largest possible digit.
                    # Since t_indices_by_digit[d_char] is sorted by index descending, the first available op_idx in the list is the latest possible operation providing this digit.
                    # This op_idx is the best choice for the current position j using an available operation.
                    best_digit_char = d_char
                    best_op_idx = op_idx
                    # We found the best possible digit (d_char) and the latest operation index (op_idx) for this position.
                    break # Stop searching for better digits for position j
                else:
                    # This operation index has already been used as the LAST update for a more significant position (i < j).
                    # Because the operations happen in order 1 to M, an operation k used as the last update for position i (i < j)
                    # cannot be the last update for position j, unless j is targeted by the same operation k later, which is impossible
                    # since we are processing positions j from left to right and deciding the final digit for j based on available ops.
                    # This op_idx cannot be the LAST update for the current position j (or any subsequent position j' > j).
                    # We discard this op_idx from future consideration for positions >= j.
                    t_indices_by_digit[d_char].pop(0)
            
            # If we found an improvement (best_op_idx != -1), we already broke the inner loop over digits
            # because we found the largest possible improving digit.

    # If we found an operation to improve s_list[j]
    if best_op_idx != -1:
        s_list[j] = best_digit_char
        used_ops[best_op_idx] = True

print("".join(s_list))