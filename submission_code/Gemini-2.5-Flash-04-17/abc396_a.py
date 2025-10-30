# Read the integer N representing the length of the sequence.
N = int(input())

# Read the sequence A as a list of integers.
# The input line is read as a string, split into space-separated parts,
# each part is converted to an integer using map(), and the result is
# converted into a list.
A = list(map(int, input().split()))

# Initialize a boolean variable to track if a triple of identical consecutive elements is found.
# Initially, we assume no such triple exists.
found_triple = False

# Iterate through the possible starting indices of a triplet.
# A triplet consists of elements at indices i, i+1, and i+2.
# The index i can range from 0 up to N-3.
# If i = N-3, the triplet is A[N-3], A[N-2], A[N-1], which are the last three elements.
# The range function range(N-2) generates integers from 0 up to N-3 (inclusive).
for i in range(N - 2):
    # Check if the element at index i, the element at index i+1,
    # and the element at index i+2 are all equal to each other.
    # Python allows chaining comparisons like A[i] == A[i+1] == A[i+2].
    if A[i] == A[i+1] == A[i+2]:
        # If they are equal, we have found a sequence of three or more identical elements.
        # Set the flag to True.
        found_triple = True
        # Since we only need to determine if *there is* such a place, we can stop searching
        # as soon as we find the first one.
        break

# After the loop finishes, check the value of the flag.
# If found_triple is True, print "Yes".
if found_triple:
    print("Yes")
# If found_triple is still False, it means the loop completed without finding a triple,
# so no such place exists in the sequence. Print "No".
else:
    print("No")