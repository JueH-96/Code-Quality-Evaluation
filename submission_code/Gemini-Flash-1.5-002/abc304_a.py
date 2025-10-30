# YOUR CODE HERE
n = int(input())
people = []
for i in range(n):
    s, a = input().split()
    people.append((s, int(a), i))

youngest_index = min(range(len(people)), key=lambda i: people[i][1])
youngest_person = people[youngest_index]

result = []
for i in range(n):
    index = (youngest_index + i) % n
    result.append(people[index][0])

for name in result:
    print(name)