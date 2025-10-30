# YOUR CODE HERE
import sys

# Read the two integer inputs A and B from standard input
# A and B represent the persons that Ringo and Snuke remember are not the culprit, respectively.
a, b = map(int, sys.stdin.readline().split())

# The suspects are persons 1, 2, and 3.
# Ringo states that person 'a' is not the culprit.
# Snuke states that person 'b' is not the culprit.

# We need to find the set of possible culprits based on these two statements.
# A person is a possible culprit if they are one of the suspects (1, 2, or 3)
# AND they are not person 'a' AND they are not person 'b'.

# Let's iterate through each suspect (1, 2, and 3) and check if they could be the culprit.
possible_culprits = []
for person in range(1, 4): # Check persons 1, 2, and 3
    # A person is a possible culprit if they are not eliminated by either witness.
    # A person is eliminated if their number is 'a' or 'b'.
    # So, a person is NOT eliminated if their number is NOT 'a' AND NOT 'b'.
    if person != a and person != b:
        possible_culprits.append(person)

# After checking all suspects, we have a list of persons who could potentially be the culprit.
# If this list contains exactly one person, then the culprit is uniquely identified.
if len(possible_culprits) == 1:
    # Print the number of the uniquely identified culprit.
    print(possible_culprits[0])
# If the list contains zero persons (which is not possible with the given constraints and logic,
# as at least one person from 1, 2, 3 must be different from 'a' and 'b' unless a=b eliminates one and another witness eliminates the others etc. Oh, wait, a=b implies two people remain)
# or more than one person (which happens when a == b, leaving two possibilities),
# the culprit cannot be uniquely identified.
else:
    # Print -1 to indicate the culprit cannot be uniquely determined.
    print(-1)