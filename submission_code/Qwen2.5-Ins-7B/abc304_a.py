# YOUR CODE HERE
n = int(input())
people = []
for i in range(n):
    s, a = input().split()
    people.append((int(a), s))
people.sort()
result = []
start = 0
for i in range(n):
    age, name = people[i]
    if i == 0:
        start = 1
    else:
        while start < n and people[start][0] != age + 1:
            start += 1
        result.append(people[start][1])
        start += 1
for name in result:
    print(name)