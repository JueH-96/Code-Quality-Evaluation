# YOUR CODE HERE

n = int(input())
people = []
for i in range(n):
    a, b = map(int, input().split())
    rate = a / (a + b)
    people.append((rate, -(a + b), i + 1))

people.sort(reverse=True)
for _, _, i in people:
    print(i, end=' ')