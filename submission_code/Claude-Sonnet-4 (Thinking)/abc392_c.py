n = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Convert to 0-indexed
P = [p - 1 for p in P]
Q = [q - 1 for q in Q]

# Create a mapping from bib number to person
bib_to_person = [0] * n
for person in range(n):
    bib = Q[person]
    bib_to_person[bib] = person

# For each bib number i (0-indexed), find the answer
S = []
for i in range(n):
    # Person wearing bib i
    person_with_bib_i = bib_to_person[i]
    # This person is staring at
    person_being_stared_at = P[person_with_bib_i]
    # The bib number of the person being stared at
    bib_of_stared_at = Q[person_being_stared_at]
    S.append(bib_of_stared_at + 1)  # Convert back to 1-indexed

print(' '.join(map(str, S)))