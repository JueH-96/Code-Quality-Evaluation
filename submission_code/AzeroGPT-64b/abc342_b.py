n = int(input())
p = list(map(int, input().split()))
q = int(input())
queries = [list(map(int, input().split())) for _ in range(q)]

# Create a dictionary to map person number to their position
person_to_position = {person: idx + 1 for idx, person in enumerate(p)}

for a, b in queries:
    print(a if person_to_position[a] < person_to_position[b] else b)