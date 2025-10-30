import sys

# Read the input string S
S = sys.stdin.readline().strip()

n = len(S)
count = 0
k = 0 # current index in S

# Iterate through the string S from left to right
# We process the string by consuming characters from the beginning (index k)
while k < n:
    # At index k, we have two options for the next button press
    # that will match the beginning of the remaining string S[k:]

    # Option 1: Use the '00' button
    # This is possible if there are at least two characters remaining (k + 1 < n)
    # AND the next two characters S[k] and S[k+1] are both '0'.
    # If this condition is met, pressing '00' is always optimal for these two characters
    # compared to pressing '0' then '0' separately (1 press vs 2 presses).
    if k + 1 < n and S[k] == '0' and S[k+1] == '0':
        # If we can press '00', we do it greedily
        count += 1 # Increment press count for the '00' button
        k += 2     # Advance the index by 2, as '00' covers two characters
    else:
        # Option 2: Use a single digit button (S[k])
        # We take this option if the '00' button cannot be used at the current position
        # (either because there aren't two characters left, or S[k:k+1] is not '00').
        # We press the button corresponding to the single digit S[k].
        # The problem guarantees S[0] is not '0', so the first digit is always 1-9.
        # Subsequent digits can be '0'. This handles all single digits.
        count += 1 # Increment press count for the single digit button
        k += 1     # Advance the index by 1, as a single digit covers one character

# The loop finishes when k reaches or exceeds n, meaning the entire string S has been displayed.
# The total count accumulated is the minimum number of button presses.
print(count)