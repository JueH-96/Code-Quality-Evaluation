import sys

def main():
    N = int(sys.stdin.readline())

    # Calculate M, the minimum number of friends.
    # Constraints are 2 <= N <= 100.
    # So N-1 >= 1, and (N-1).bit_length() >= 1. Thus M >= 1.
    M = (N - 1).bit_length()
        
    print(M, flush=True)

    # For each friend, describe the bottles they drink.
    # This loop runs M times. Since M >= 1, it always runs at least once.
    for p_one_indexed in range(1, M + 1):
        # p_one_indexed is the 1-indexed friend number (1 to M)
        # This friend will test the (p_one_indexed - 1)-th bit.
        friend_tests_bit_idx = p_one_indexed - 1 
        
        bottles_for_this_friend = []
        # Iterate through all bottles (1-indexed)
        for bottle_num_one_indexed in range(1, N + 1):
            # Convert to 0-indexed for bitwise operations
            bottle_idx_zero_indexed = bottle_num_one_indexed - 1
            
            # Check if the (friend_tests_bit_idx)-th bit of bottle_idx_zero_indexed is 1
            if (bottle_idx_zero_indexed >> friend_tests_bit_idx) & 1:
                bottles_for_this_friend.append(bottle_num_one_indexed)
        
        # Print K_i (count of bottles for friend i) followed by bottle numbers.
        # Bottles are added in ascending order by iterating j from 1 to N,
        # so no explicit sort is needed for bottles_for_this_friend.
        # bottles_for_this_friend will not be empty:
        # Friend testing bit B (0-indexed friend_tests_bit_idx) will test bottle 2^B + 1,
        # because 2^B has bit B set. Since B < M, we know 2^B <= 2^(M-1).
        # And since M = (N-1).bit_length(), we have 2^(M-1) <= N-1.
        # So bottle 2^B + 1 is within the range [1, N].
        print(len(bottles_for_this_friend), *bottles_for_this_friend, flush=True)

    # Read the results string S from the judge. S will have length M.
    S = sys.stdin.readline().strip()

    spoiled_bottle_idx_zero_indexed = 0
    # Decode S to find the 0-indexed spoiled bottle.
    # This loop runs M times.
    for k in range(M): # k is the 0-indexed bit position
        # S is 0-indexed in Python. S[k] is the outcome for friend k+1 (1-indexed).
        # Friend k+1 tested bit k.
        # If S[k] is '1', it means friend k+1 got sick, so bit k of the spoiled bottle's index is 1.
        if S[k] == '1':
            spoiled_bottle_idx_zero_indexed += (1 << k) # Add 2^k
    
    # The actual bottle number is 1 greater than the 0-indexed version.
    print(spoiled_bottle_idx_zero_indexed + 1, flush=True)

if __name__ == '__main__':
    main()