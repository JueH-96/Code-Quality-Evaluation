# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Create mapping from bib number to person number
bib_to_person = {}
for i in range(N):
    bib_to_person[Q[i]] = i + 1

# For each bib number, find the answer
result = []
for bib_num in range(1, N + 1):
    # Find which person wears this bib
    person = bib_to_person[bib_num]
    # Find which person they stare at (person is 1-indexed, P is 0-indexed)
    stared_at = P[person - 1]
    # Find which bib the stared-at person wears
    bib_of_stared = Q[stared_at - 1]
    result.append(bib_of_stared)

print(' '.join(map(str, result)))