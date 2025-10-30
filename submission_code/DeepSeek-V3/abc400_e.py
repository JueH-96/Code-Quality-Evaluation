n = int(input())
P = list(map(int, input().split()))
people = [(P[i], i) for i in range(n)]
rank = [0] * n
current_rank = 1

# We need to process in descending order of P_i, handling ties
# Create a list sorted in descending order, then process each group
sorted_people = sorted(people, key=lambda x: (-x[0], x[1]))

i = 0
while i < n:
    current_score = sorted_people[i][0]
    # Find all entries with current_score
    j = i
    while j < n and sorted_people[j][0] == current_score:
        j += 1
    # The group is from i to j-1
    k = j - i
    for idx in range(i, j):
        original_pos = sorted_people[idx][1]
        rank[original_pos] = current_rank
    current_rank += k
    i = j

for r in rank:
    print(r)