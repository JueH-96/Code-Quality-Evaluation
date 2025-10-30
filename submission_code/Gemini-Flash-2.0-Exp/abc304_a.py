N = int(input())
people = []
for i in range(N):
    S, A = input().split()
    people.append((S, int(A)))

youngest_index = 0
youngest_age = float('inf')
for i in range(N):
    if people[i][1] < youngest_age:
        youngest_age = people[i][1]
        youngest_index = i

for i in range(N):
    index = (youngest_index + i) % N
    print(people[index][0])