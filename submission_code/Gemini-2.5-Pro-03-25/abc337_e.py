# YOUR CODE HERE
import sys

def solve():
    # Read the number of bottles N from standard input
    N = int(sys.stdin.readline())

    # Calculate the minimum number of friends M required.
    # The minimum M must satisfy 2^M >= N.
    # This is equivalent to M = ceil(log2(N)).
    # For integers N >= 2, this value M can be efficiently computed using the bit_length() method 
    # of the integer N-1. (N-1).bit_length() gives the minimum number of bits required to represent N-1,
    # which is exactly the M we need.
    # Example: N=8. N-1=7. Binary of 7 is 111. bit_length() is 3. M=3. 2^3 = 8 >= 8. Correct.
    # Example: N=9. N-1=8. Binary of 8 is 1000. bit_length() is 4. M=4. 2^4 = 16 >= 9. Correct.
    # The problem constraint N >= 2 ensures N-1 >= 1, so bit_length() is always >= 1.
    M = (N - 1).bit_length()

    # Print the minimum number of friends M, followed by a newline.
    # Flush the output buffer to ensure the judge receives the output immediately.
    print(M, flush=True)

    # Prepare the lists of bottles that each friend will drink.
    # We will have M lists, one for each friend (indexed 1 to M).
    friend_bottles = [] 
    
    # We use 0-based index `i` internally for iterations (0 to M-1).
    # This index `i` corresponds to the i-th bit position (0-indexed) in the binary representation.
    # It also corresponds to friend number `i+1` (1-based index).
    for i in range(M): 
        bottles_for_this_friend = []
        # Iterate through all bottle numbers X from 1 to N.
        for X in range(1, N + 1):
            # The core strategy is based on binary representation. Each bottle X is associated with 
            # the M-bit binary representation of X-1.
            # Friend `i+1` (associated with bit position `i`) drinks bottle X if and only if 
            # the i-th bit of the number X-1 is 1.
            # We check the i-th bit using a right shift `>>` and bitwise AND `&`.
            # `(X - 1) >> i` shifts the bits of X-1 right by i positions. The i-th bit becomes the 0-th bit.
            # `& 1` extracts the value of this 0-th bit (which is either 0 or 1).
            if ((X - 1) >> i) & 1:
                # If the i-th bit is 1, friend `i+1` drinks bottle X. Add X to this friend's list.
                bottles_for_this_friend.append(X)
        
        # Store the generated list of bottles for friend `i+1`.
        friend_bottles.append(bottles_for_this_friend)

    # Print the distribution plan for each friend.
    # Iterate M times, once for each friend from 1 to M.
    for i in range(M):
        # `friend_bottles[i]` contains the list of bottles for friend `i+1`.
        current_friend_bottles = friend_bottles[i]
        
        # Print the number of bottles K_i this friend drinks.
        # Then print the bottle numbers A_{i,1}, A_{i,2}, ..., A_{i,K_i}.
        # The list `current_friend_bottles` is guaranteed to contain bottle numbers in ascending order
        # because we iterated through X from 1 to N when adding them.
        # Use the `*` operator to unpack the list elements as separate arguments to `print`.
        print(len(current_friend_bottles), *current_friend_bottles, flush=True)

    # After printing all distributions, read the result string S from the judge.
    # S is a string of length M consisting of '0's and '1's.
    # S[i] ('0'-indexed) indicates the status of friend `i+1` ('1'-based).
    S = sys.stdin.readline().strip()

    # Decode the result string S to determine the number of the spoiled bottle X.
    # If S[i] is '1', it means friend `i+1` got sick.
    # According to our strategy, friend `i+1` got sick if and only if they drank the spoiled bottle X.
    # This happens if and only if the i-th bit of X-1 is 1.
    # Therefore, the string S represents the bits of X-1. S[0] is the 0-th bit, S[1] is the 1st bit, ..., S[M-1] is the (M-1)-th bit.
    # We can reconstruct the value X-1 by summing powers of 2 based on the '1's in S.
    spoiled_bottle_val_minus_1 = 0
    for i in range(M):
        # Check the i-th character of S (corresponding to friend i+1).
        if S[i] == '1':
            # If the character is '1', it means the i-th bit of X-1 is 1.
            # Add 2^i to the reconstructed value. `(1 << i)` computes 2^i efficiently.
            spoiled_bottle_val_minus_1 += (1 << i)
            # Alternatively, using bitwise OR: `spoiled_bottle_val_minus_1 |= (1 << i)` also works.

    # The reconstructed value `spoiled_bottle_val_minus_1` is equal to X-1.
    # Add 1 to get the final answer, which is the 1-based bottle number X.
    spoiled_bottle = spoiled_bottle_val_minus_1 + 1
    
    # Print the identified number of the spoiled bottle X'.
    # This must be the final output before terminating.
    print(spoiled_bottle, flush=True)

# Execute the main logic of the program by calling the solve function.
solve()

# The program terminates automatically after the final print statement.