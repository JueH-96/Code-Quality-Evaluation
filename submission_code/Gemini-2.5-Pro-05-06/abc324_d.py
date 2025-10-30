import collections

def main():
    N = int(input())
    S_input_str = input() # Input string S

    # Count frequencies of digits in the input string S_input_str
    counts_S = collections.Counter(S_input_str)
    
    found_squares = set()
    
    # Iterate through integers i, calculate square_val = i*i
    # The loop starts from i = 0.
    i = 0
    while True:
        square_val = i * i
        s_square_val = str(square_val) # String representation of the current square value
        len_s_square_val = len(s_square_val)

        if len_s_square_val > N:
            # If the square number has more digits than N,
            # it cannot be formed by a permutation of S_input_str (which has N digits).
            # All subsequent squares will also be longer, so we can stop.
            break

        # Check if this square_val can be formed by a permutation of S_input_str.
        # The number `square_val` (represented by `s_square_val`) has `len_s_square_val` digits.
        # To form it using N digits from S_input_str, the permutation of S_input_str
        # must effectively be `N - len_s_square_val` leading zeros,
        # followed by the `len_s_square_val` digits of `s_square_val`.
        
        # Count frequencies of digits in s_square_val
        counts_X = collections.Counter(s_square_val)
        
        # Calculate the number of padding zeros required. These are zeros from S_input_str
        # that would act as leading zeros in the N-digit representation.
        num_padding_zeros = N - len_s_square_val
        
        # Construct the target frequency map for the N digits.
        # This map includes digits from s_square_val and the padding zeros.
        target_counts_for_S = counts_X.copy() # Start with digits from s_square_val
        # Add the count for padding zeros to the count of '0'.
        # .get('0', 0) handles cases where s_square_val itself has no zeros.
        target_counts_for_S['0'] = target_counts_for_S.get('0', 0) + num_padding_zeros
        
        # The sum of frequencies in target_counts_for_S will always be N.
        # (len_s_square_val - count of '0's in s_square_val) + (count of '0's in s_square_val + num_padding_zeros)
        # = len_s_square_val + num_padding_zeros = len_s_square_val + (N - len_s_square_val) = N.

        # Check if the multiset of digits in S_input_str matches this target multiset
        if counts_S == target_counts_for_S:
            found_squares.add(square_val)
            
        i += 1

    print(len(found_squares))

if __name__ == '__main__':
    main()