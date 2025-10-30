import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Read N from the first element
N = int(data[0])

# Read P_list from data[1] to data[N]
P_list = [int(x) for x in data[1:N+1]]

# Read Q_list from data[N+1] to data[2*N]
Q_list = [int(x) for x in data[N+1:2*N+1]]

# Create stare_at list (1-based)
stare_at = [0] * (N + 1)
for person in range(1, N + 1):
    stare_at[person] = P_list[person - 1]

# Create bib_of_person list (1-based)
bib_of_person = [0] * (N + 1)
for person in range(1, N + 1):
    bib_of_person[person] = Q_list[person - 1]

# Create bib_to_person list (1-based)
bib_to_person = [0] * (N + 1)
for person in range(1, N + 1):
    bib = Q_list[person - 1]  # bib number for this person
    bib_to_person[bib] = person

# Compute S for each bib number from 1 to N
result = []
for bib_num in range(1, N + 1):
    person_x = bib_to_person[bib_num]  # Person with bib bib_num
    person_y = stare_at[person_x]     # Person that person_x stares at
    s_i = bib_of_person[person_y]    # Bib number of person_y
    result.append(str(s_i))

# Output the result separated by spaces
print(' '.join(result))