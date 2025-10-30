# YOUR CODE HERE

N = int(input())
people = [input().split() for _ in range(N)]
people.sort(key=lambda x: int(x[1]))

for person in people:
    print(person[0])