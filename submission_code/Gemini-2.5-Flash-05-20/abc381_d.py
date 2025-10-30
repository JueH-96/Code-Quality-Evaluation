import sys
from collections import Counter

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    max_len = 0
    left = 0
    # freq stores counts of values that appear as the *first* element of a pair (X_k)
    # within the current window A[left:right+1].
    # Example: for a sequence (1,1,2,2), freq would be {1:1, 2:1}.
    # If a value appears multiple times as a first element (e.g., (1,1,2,2,1,1)),
    # it violates the distinctness condition, and the window must be shrunk.
    freq = Counter()

    for right in range(N):
        current_val = A[right]

        if (right - left) % 2 == 0:
            # A[right] is at an 'even' offset from `left`, meaning it's expected
            # to be the first element of a pair (e.g., A[left], A[left+2], etc.)
            freq[current_val] += 1
            
            # If this 'first element' value (current_val) has already appeared
            # as a first element in a previous pair within the current window [left, right],
            # it violates the condition that all x_k values must be distinct.
            # We must shrink the window from the left until this condition is met.
            while freq[current_val] > 1:
                # Remove the pair (A[left], A[left+1]) from consideration.
                # A[left] was the first element of this pair.
                freq[A[left]] -= 1 # Decrement count for A[left] (which was a "first element")
                left += 2         # Move `left` pointer to effectively remove the pair
        else:
            # A[right] is at an 'odd' offset from `left`, meaning it's expected
            # to be the second element of a pair (i.e., it should match A[right-1]).
            if current_val != A[right-1]:
                # The condition A[X_2i-1] == A[X_2i] is violated.
                # The current sequence A[left:right+1] is not a 1122 sequence.
                # A new potential 1122 sequence must start from `right`.
                left = right      # Set the new start of the window to `right`
                freq.clear()      # Clear the frequency map as we are starting a completely new sequence
                
                # A[right] (which is now also A[left]) is the first element of this new single-element window.
                # Add it to the frequency map.
                freq[current_val] += 1
                # No need for `while freq[current_val] > 1` check here, because `freq` was just cleared,
                # ensuring `current_val` is the only entry (with count 1) for this new window.
                
        # After processing A[right] and potentially adjusting `left`:
        # If the current window A[left:right+1] has an even length, it means
        # it *could* be a valid 1122 sequence. The other conditions (pair equality
        # and distinct x_k values) are implicitly maintained by the logic above.
        if (right - left + 1) % 2 == 0:
            max_len = max(max_len, right - left + 1)

    print(max_len)

solve()