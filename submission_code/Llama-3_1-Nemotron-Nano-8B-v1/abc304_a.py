n = int(input())

people = []
for i in range(n):
    s, a = input().split()
    people.append((int(a), s, i + 1))

youngest = min(people, key=lambda x: x[0])
start = youngest[2]

index_to_person = {p[2]: p for p in people}

order = []
for i in range(n):
    original_index = (start - 1 + i) % n + 1
    order.append(original_index)

for idx in order:
    print(index_to_person[idx][1])