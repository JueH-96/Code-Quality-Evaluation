# YOUR CODE HERE
n = int(input())
people = list(map(int, input().split()))
q = int(input())
for _ in range(q):
    a, b = map(int, input().split())
    print(people[a-1] if people[a-1] < people[b-1] else people[b-1])