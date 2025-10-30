# YOUR CODE HERE

n = int(input())
people = []
for _ in range(n):
    name, age = input().split()
    people.append((name, int(age)))

people.sort(key=lambda x: x[1])

youngest_age = people[0][1]
youngest_index = None
for i in range(n):
    if people[i][1] == youngest_age:
        youngest_index = i
        break

for i in range(youngest_index, youngest_index + n):
    print(people[i % n][0])