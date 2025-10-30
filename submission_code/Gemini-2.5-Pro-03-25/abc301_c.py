# YOUR CODE HERE
import sys
from collections import Counter

# Function to solve the problem
def solve():
    # Read input strings S and T from standard input
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Calculate character counts for both strings using collections.Counter
    countS = Counter(S)
    countT = Counter(T)

    # Define the set of special characters that '@' can be replaced with
    atcoder_chars = set(['a', 't', 'c', 'o', 'd', 'e', 'r'])

    # Check Condition 1: Counts of non-special, non-'@' characters must match exactly.
    # Since these characters cannot be changed and cannot be formed from '@', their counts
    # must be identical in both strings S and T for it to be possible to make them identical eventually.
    possible = True
    # Iterate through all lowercase English letters 'a' through 'z'
    for char_ord in range(ord('a'), ord('z') + 1):
        char = chr(char_ord)
        # Check if the character is a "normal" letter:
        # It must not be '@' and it must not be one of the characters '@' can turn into.
        if char not in atcoder_chars and char != '@': 
            # If the counts of this normal character differ between S and T, it's impossible to win.
            # Note: Accessing Counter with a non-existent key returns 0, which correctly handles characters
            # not present in one or both strings.
            if countS[char] != countT[char]:
                possible = False
                break  # No need to check further characters if impossibility is found
    
    # If Condition 1 failed (found a mismatch in counts of a normal letter), print "No" and exit.
    if not possible:
        print("No")
        return

    # Calculate the minimum required '@' replacements for each string to balance the counts
    # of ATCODER characters.
    # reqS: minimum number of '@' symbols S must use to potentially match T's counts for ATCODER characters.
    # reqT: minimum number of '@' symbols T must use to potentially match S's counts for ATCODER characters.
    reqS = 0 
    reqT = 0 

    # Iterate through the set of characters that '@' can be replaced with ('a', 't', 'c', 'o', 'd', 'e', 'r')
    for char in atcoder_chars:
        # Calculate the difference in counts for the current ATCODER character: T's count minus S's count.
        diff = countT[char] - countS[char] 
        
        if diff > 0:
            # If T initially has more instances of 'char' than S (diff > 0), then S needs to compensate
            # by replacing at least 'diff' of its '@' symbols with 'char'.
            # This requirement contributes to the total minimum '@' needed by S (reqS).
            reqS += diff 
        elif diff < 0: # Equivalent to countS[char] > countT[char]
            # If S initially has more instances of 'char' than T (diff < 0), then T needs to compensate
            # by replacing at least '-diff' (which is positive) of its '@' symbols with 'char'.
            # This requirement contributes to the total minimum '@' needed by T (reqT).
            reqT += -diff # Note: -diff is positive because diff is negative.
        # If diff == 0, the counts for 'char' are initially equal, so no minimum '@' replacement
        # is mandated for this character based on initial counts.

    # Get the actual counts of '@' symbols available in S and T.
    # Accessing Counter['@'] correctly returns 0 if '@' is not present.
    num_atS = countS['@']
    num_atT = countT['@']

    # Check Conditions 2 and 3: Ensure each string has enough '@' symbols to meet its minimum requirements.
    # S must have at least reqS '@' symbols to cover its deficiencies for ATCODER characters relative to T.
    # T must have at least reqT '@' symbols to cover its deficiencies for ATCODER characters relative to S.
    if num_atS >= reqS and num_atT >= reqT:
        # If both strings have enough '@' symbols to potentially balance the counts of ATCODER characters,
        # combined with the fact that non-special characters already match (Condition 1), it is possible to win.
        print("Yes")
    else:
        # If either string does not have enough '@' symbols to meet its minimum requirement,
        # it's impossible to make the strings identical.
        print("No")

# Call the solve function to execute the logic when the script runs
solve()
# END OF YOUR CODE HERE