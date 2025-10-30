N = int(input())
people = []
for _ in range(N):
    name, age = input().split()
    age = int(age)
    people.append((name, age))

people.sort(key=lambda x: x[1])
youngest = people[0][0]
people = people[1:] + [people[0]]

start = 0
for i in range(N):
    if people[i][0] == youngest:
        start = i
        break

for i in range(N):
    print(people[(start+i)%N][0])