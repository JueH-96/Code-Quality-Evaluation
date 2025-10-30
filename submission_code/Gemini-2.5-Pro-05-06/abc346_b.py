import sys

def main():
    W, B = map(int, sys.stdin.readline().split())

    piano_pattern = "wbwbwwbwbwbw"
    
    # The infinite string S is formed by repeating piano_pattern.
    # Max length of target substring is W_max + B_max = 100 + 100 = 200.
    # Length of piano_pattern is 12.
    # We need to be able to extract a substring of length up to 200,
    # starting at an offset up to 11 (from index 0 to 11 of a pattern block).
    # The maximum end index (exclusive) for slicing would be offset + length.
    # Max offset = 11. Max length = 200.
    # So, a slice could be s_test[11 : 11+200] = s_test[11 : 211].
    # This means s_test must have length at least 211.
    #
    # piano_pattern * K. Length of piano_pattern is 12.
    # Minimum K such that 12 * K >= 211 is K = ceil(211/12) = ceil(17.58...) = 18.
    # So, 18 repetitions (length 18 * 12 = 216) is the minimum needed.
    # Using K=20 (length 240) is simple and safely sufficient.
    
    s_test = piano_pattern * 20 
    
    target_len = W + B

    # Constraint: W+B >= 1, so target_len is at least 1.

    # Iterate through all 12 possible start offsets within a piano_pattern block.
    for i in range(len(piano_pattern)): # i ranges from 0 to 11
        # Extract the candidate substring.
        # It starts at index i in s_test and has length target_len.
        sub = s_test[i : i + target_len]
        
        # Since s_test is long enough (length 240) and the maximum value of
        # (i + target_len) is (11 + 200) = 211, the slice s_test[i : i + target_len]
        # will always produce a substring of exactly target_len characters.
        
        # Count 'w's in this substring.
        w_count = sub.count('w')
        
        # Check if this count matches the required W.
        # If w_count == W, then b_count (number of 'b's) must be target_len - w_count.
        # Since target_len = W + B, this means b_count = (W + B) - W = B.
        # So, checking w_count == W is sufficient.
        if w_count == W:
            print("Yes")
            return # Found a valid substring, so we can stop.
            
    # If the loop finishes, it means no such substring was found
    # starting at any of the 12 distinct offsets.
    print("No")

if __name__ == '__main__':
    main()