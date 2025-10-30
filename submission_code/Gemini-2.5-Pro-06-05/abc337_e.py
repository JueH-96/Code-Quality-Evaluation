import sys

def solve():
    """
    This program solves the interactive spoiled juice problem by using a binary encoding scheme.
    """
    # It's an interactive problem, so we must flush the output buffer after each print.
    
    # Read N, the total number of bottles.
    try:
        n = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handle cases where input reading might fail.
        return

    # --- Part 1: Determine the number of friends and the juice distribution ---

    # Calculate M, the minimum number of friends required.
    # We need to distinguish between N possibilities. With M friends, we have 2^M outcomes.
    # So, we need 2^M >= N. The smallest integer M is ceil(log2(N)).
    # For N > 1, this can be efficiently calculated as (N-1).bit_length().
    if n > 1:
        m = (n - 1).bit_length()
    else:
        # This case is excluded by constraints (N>=2), but for completeness,
        # N=1 would require 0 friends.
        m = 0

    # Print the number of friends.
    print(m)
    sys.stdout.flush()

    # The core strategy is to map each bottle `j` to the binary representation of `j-1`.
    # Friend `i` (0-indexed) corresponds to bit `i`. They drink from bottle `j`
    # if the `i`-th bit of `j-1` is set.
    if m > 0:
        # Prepare lists of bottles for each friend.
        friend_bottles = [[] for _ in range(m)]

        for bottle_num in range(1, n + 1):
            val_to_encode = bottle_num - 1
            for friend_idx in range(m):
                # Check if the friend_idx'th bit is set.
                if (val_to_encode >> friend_idx) & 1:
                    friend_bottles[friend_idx].append(bottle_num)

        # Print the distribution for each friend.
        for bottles_for_friend in friend_bottles:
            # The bottle numbers are already in ascending order.
            output_line = [len(bottles_for_friend)] + bottles_for_friend
            print(*output_line)
            sys.stdout.flush()

    # --- Part 2: Read results and determine the spoiled bottle ---

    # Read the sickness report string S from the judge.
    s = sys.stdin.readline().strip()

    # If the judge returns -1, it means our previous output was incorrect.
    if s == "-1":
        return

    # Decode the binary string S to find the original value.
    # S[i] corresponds to the i-th friend, which we mapped to the i-th bit.
    decoded_val = 0
    for i in range(len(s)):
        if s[i] == '1':
            decoded_val += (1 << i)

    # The decoded value is `spoiled_bottle_num - 1`. Add 1 to get the answer.
    # If S is all '0's, decoded_val is 0, and the answer is 1. This correctly
    # identifies bottle 1, which corresponds to the value 0 (binary 00...0).
    spoiled_bottle = decoded_val + 1

    print(spoiled_bottle)
    sys.stdout.flush()

solve()