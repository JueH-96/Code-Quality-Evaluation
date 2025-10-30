import sys
import functools

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
L = int(data[index + 2])
index += 3
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + M]))
index += M
C = list(map(int, data[index:index + L]))

# Make sorted tuples for initial state
initial_tak_hand = tuple(sorted(A))
initial_aok_hand = tuple(sorted(B))
initial_table = tuple(sorted(C))

# Define dp with memoization using lru_cache
@functools.lru_cache(None)
def dp(tak_hand, aok_hand, table, turn):
    if turn == 0:  # Takahashi's turn
        if not tak_hand:  # No cards, lose
            return False
        # Iterate over unique values in tak_hand
        for X in set(tak_hand):
            # Create new tak_hand without one X
            tak_hand_list_noX = list(tak_hand)
            idx_X = tak_hand_list_noX.index(X)
            del tak_hand_list_noX[idx_X]
            new_tak_hand_noX = tuple(tak_hand_list_noX)
            # Not take action
            new_table_withX = tuple(sorted(list(table) + [X]))
            if not dp(new_tak_hand_noX, aok_hand, new_table_withX, 1 - turn):
                return True
            # Take Y actions
            for Y in set(table):
                if Y < X:
                    # New tak_hand with Y added
                    tak_hand_take_list = list(new_tak_hand_noX)
                    tak_hand_take_list.append(Y)
                    new_tak_hand_take = tuple(sorted(tak_hand_take_list))
                    # New table with Y removed and X added
                    table_take_list = list(table)
                    idx_Y = table_take_list.index(Y)
                    del table_take_list[idx_Y]
                    table_take_list.append(X)
                    new_table_take = tuple(sorted(table_take_list))
                    if not dp(new_tak_hand_take, aok_hand, new_table_take, 1 - turn):
                        return True
        return False
    else:  # Aoki's turn
        if not aok_hand:  # No cards, lose
            return False
        # Iterate over unique values in aok_hand
        for X in set(aok_hand):
            # Create new aok_hand without one X
            aok_hand_list_noX = list(aok_hand)
            idx_X = aok_hand_list_noX.index(X)
            del aok_hand_list_noX[idx_X]
            new_aok_hand_noX = tuple(aok_hand_list_noX)
            # Not take action
            new_table_withX = tuple(sorted(list(table) + [X]))
            if not dp(tak_hand, new_aok_hand_noX, new_table_withX, 1 - turn):
                return True
            # Take Y actions
            for Y in set(table):
                if Y < X:
                    # New aok_hand with Y added
                    aok_hand_take_list = list(new_aok_hand_noX)
                    aok_hand_take_list.append(Y)
                    new_aok_hand_take = tuple(sorted(aok_hand_take_list))
                    # New table with Y removed and X added
                    table_take_list = list(table)
                    idx_Y = table_take_list.index(Y)
                    del table_take_list[idx_Y]
                    table_take_list.append(X)
                    new_table_take = tuple(sorted(table_take_list))
                    if not dp(tak_hand, new_aok_hand_take, new_table_take, 1 - turn):
                        return True
        return False

# Compute the result with initial state, Takahashi starts
result = dp(initial_tak_hand, initial_aok_hand, initial_table, 0)

# Output the winner
if result:
    print("Takahashi")
else:
    print("Aoki")