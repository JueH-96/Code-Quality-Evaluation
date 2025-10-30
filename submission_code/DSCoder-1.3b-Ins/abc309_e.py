# YOUR CODE HERE

N, M = map(int, input().split())
parents = list(map(int, input().split()))
insurance_data = [list(map(int, input().split())) for _ in range(M)]

covered_people = set()

for i in range(M):
    person, times = insurance_data[i]
    for _ in range(times):
        covered_people.add(person)
        person = parents[person]

print(len(covered_people))