import sys
import math

def solve():
    # Read N, the number of bottles.
    N = int(sys.stdin.readline())

    # Calculate M, the minimum number of friends needed.
    # M is the smallest integer such that 2^M >= N.
    # This is equivalent to M = ceil(log2(N)).
    # We use 0-indexed bottles internally (0 to N-1) for binary representation.
    # So we need M bits to represent numbers from 0 to N-1.
    if N == 1: # Constraints say N >= 2, but for completeness or edge cases in log
        M = 0
    else:
        M = math.ceil(math.log2(N))
    
    # Print M and flush stdout to ensure the judge receives the output.
    sys.stdout.write(str(M) + '
')
    sys.stdout.flush()

    # Prepare a list of lists to store which bottles each friend will drink.
    # friend_bottles[k] will store bottles for the (k+1)-th friend,
    # corresponding to the k-th bit (0-indexed from the right, LSB).
    friend_bottles = [[] for _ in range(M)]

    # Iterate through each bottle from 1 to N.
    for j in range(1, N + 1):
        # Map the 1-indexed bottle 'j' to its 0-indexed equivalent 'j_0_indexed'.
        j_0_indexed = j - 1
        
        # For each bit position 'k' from 0 to M-1 (representing friends 1 to M).
        for k in range(M):
            # Check if the k-th bit of 'j_0_indexed' is 1.
            # (j_0_indexed >> k) shifts the number right by k bits,
            # ( & 1) checks if the rightmost bit is 1.
            if (j_0_indexed >> k) & 1:
                # If the k-th bit is 1, friend (k+1) should drink this bottle 'j'.
                friend_bottles[k].append(j)
    
    # Print the distribution for each friend.
    for i in range(M):
        bottles_for_current_friend = friend_bottles[i]
        K_i = len(bottles_for_current_friend)
        
        # Format the output as "K_i A_i_1 A_i_2 ...".
        # Bottle numbers are naturally in ascending order because we iterated 'j' from 1 to N.
        output_line = f"{K_i} {' '.join(map(str, bottles_for_current_friend))}"
        sys.stdout.write(output_line + '
')
        sys.stdout.flush()

    # Read the stomach upset information string 'S'.
    # S[k] is '1' if friend (k+1) got sick, '0' otherwise.
    S = sys.stdin.readline().strip()

    # Reconstruct the 0-indexed spoiled bottle number from the string 'S'.
    # S[k] corresponds to the k-th bit (LSB).
    spoiled_0_indexed = 0
    for k in range(M):
        if S[k] == '1':
            # If the k-th friend got upset, it means the k-th bit of the
            # 0-indexed spoiled bottle number is 1.
            spoiled_0_indexed += (1 << k) # Add 2^k to the result

    # Convert the 0-indexed spoiled bottle number back to 1-indexed.
    spoiled_bottle_number = spoiled_0_indexed + 1

    # Print the identified spoiled bottle number and terminate.
    sys.stdout.write(str(spoiled_bottle_number) + '
')
    sys.stdout.flush()

# Call the solve function to execute the program logic.
solve()