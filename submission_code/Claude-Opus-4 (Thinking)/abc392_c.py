# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Create a mapping from bib number to person index (0-indexed)
bib_to_person_idx = {}
for i in range(N):
    bib_to_person_idx[Q[i]] = i

# For each bib number i from 1 to N
S = []
for i in range(1, N + 1):
    # Find the person index wearing bib i
    person_idx = bib_to_person_idx[i]
    # Find who this person is staring at (convert to 0-indexed)
    target_idx = P[person_idx] - 1
    # Find the bib number of the target
    S.append(Q[target_idx])

print(' '.join(map(str, S)))