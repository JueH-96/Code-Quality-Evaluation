n = int(input())
p_list = list(map(int, input().split()))
people = []
for i in range(n):
    people.append((p_list[i], i + 1))
people.sort(key=lambda x: (-x[0], x[1]))

ranks = [0] * (n + 1)  # 1-based indexing

if n == 0:
    pass  # As per constraints, n >= 1, so this is redundant but safe
else:
    current_score = people[0][0]
    start = 0
    r = 1
    for i in range(1, len(people)):
        if people[i][0] != current_score:
            # Process the previous group
            group_size = i - start
            for k in range(start, i):
                idx = people[k][1]
                ranks[idx] = r
            r += group_size
            # Update current score and start of next group
            current_score = people[i][0]
            start = i
    # Process the last group
    group_size = len(people) - start
    for k in range(start, len(people)):
        idx = people[k][1]
        ranks[idx] = r

for i in range(1, n + 1):
    print(ranks[i])