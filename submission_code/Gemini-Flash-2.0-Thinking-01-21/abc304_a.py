def solve():
    n = int(input())
    people = []
    for _ in range(n):
        s, a = input().split()
        people.append({'name': s, 'age': int(a)})

    youngest_person_index = 0
    min_age = people[0]['age']
    for i in range(1, n):
        if people[i]['age'] < min_age:
            min_age = people[i]['age']
            youngest_person_index = i

    clockwise_names = []
    current_index = youngest_person_index
    for _ in range(n):
        clockwise_names.append(people[current_index]['name'])
        current_index = (current_index + 1) % n

    for name in clockwise_names:
        print(name)

if __name__ == "__main__":
    solve()