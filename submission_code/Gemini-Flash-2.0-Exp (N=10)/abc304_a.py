def solve():
    n = int(input())
    people = []
    for i in range(n):
        s, a = input().split()
        people.append((s, int(a), i))
    
    youngest_person = min(people, key=lambda x: x[1])
    youngest_index = youngest_person[2]
    
    result = []
    for i in range(n):
        current_index = (youngest_index + i) % n
        result.append(people[current_index][0])
    
    for name in result:
        print(name)

solve()