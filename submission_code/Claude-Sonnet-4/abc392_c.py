# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Create mapping from bib number to person number
bib_to_person = {}
for person in range(1, N + 1):
    bib_number = Q[person - 1]
    bib_to_person[bib_number] = person

# For each bib number i, find the answer
answers = []
for i in range(1, N + 1):
    # Find person wearing bib i
    person_with_bib_i = bib_to_person[i]
    
    # Find who this person is staring at
    stared_at_person = P[person_with_bib_i - 1]
    
    # Find what bib the stared-at person is wearing
    stared_at_bib = Q[stared_at_person - 1]
    
    answers.append(stared_at_bib)

print(' '.join(map(str, answers)))