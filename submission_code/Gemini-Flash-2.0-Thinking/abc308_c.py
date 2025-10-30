def solve():
    n = int(input())
    people_data = []
    for i in range(1, n + 1):
        a, b = map(int, input().split())
        people_data.append((i, a, b))

    def get_success_rate(a, b):
        return a / (a + b)

    def sort_key(item):
        person_id, a, b = item
        rate = get_success_rate(a, b)
        return (-rate, person_id)

    sorted_people_data = sorted(people_data, key=sort_key)
    result = [person_id for person_id, _, _ in sorted_people_data]
    print(*result)

solve()