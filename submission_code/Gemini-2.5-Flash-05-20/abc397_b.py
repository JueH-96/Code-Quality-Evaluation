# YOUR CODE HERE
import sys

def solve():
    S = sys.stdin.readline().strip()

    insertions = 0
    # `expected_char` represents the character we are currently looking for in the pattern ('i' or 'o').
    # It alternates between 'i' and 'o' as we successfully match/insert characters.
    expected_char = 'i' 

    for s_char in S:
        if s_char == expected_char:
            # If the character from S matches what we expect in the pattern,
            # we consume it and move to the next expected character in the pattern.
            # This means `expected_char` simply toggles.
            if expected_char == 'i':
                expected_char = 'o'
            else: # expected_char == 'o'
                expected_char = 'i'
        else:
            # If the character from S does NOT match what we expect in the pattern:
            # 1. We must INSERT the `expected_char` to maintain the pattern.
            insertions += 1
            
            # 2. After inserting, the pattern now expects the *next* character.
            #    So, `expected_char` toggles.
            if expected_char == 'i':
                expected_char = 'o'
            else: # expected_char == 'o'
                expected_char = 'i'
            
            # 3. Now, the current `s_char` (which caused the mismatch) must match this *new* `expected_char`.
            #    (Example: `expected_char` was 'i', `s_char` was 'o'. We inserted 'i'. `expected_char` became 'o'.
            #    Now `s_char` ('o') matches the new `expected_char` ('o')).
            #    Since `s_char` is now consumed, `expected_char` toggles *again* for the next slot in the pattern.
            if expected_char == 'i':
                expected_char = 'o'
            else: # expected_char == 'o'
                expected_char = 'i'
    
    # After iterating through all characters in S, we have conceptually formed a sequence
    # by using all characters from S and inserting `insertions` characters.
    # The total length of this sequence is `len(S) + insertions`.
    
    # The problem requires the final string to have an even length.
    # If the current total length is odd, we need one more insertion to make it even.
    if (len(S) + insertions) % 2 != 0:
        insertions += 1
    
    print(insertions)

solve()