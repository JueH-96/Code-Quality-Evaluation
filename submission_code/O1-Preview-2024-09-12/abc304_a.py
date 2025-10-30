# YOUR CODE HERE
N = int(input())
people = []
for _ in range(N):
    S_i, A_i = input().split()
    people.append((S_i, int(A_i)))

ages = [person[1] for person in people]
min_age = min(ages)
min_index = ages.index(min_age)

for i in range(N):
    index = (min_index + i) % N
    print(people[index][0])