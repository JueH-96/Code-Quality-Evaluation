def solve():
    n, m = map(int, input().split())
    events = []
    for _ in range(m):
        events.append(list(map(int, input().split())))

    current_row = list(range(1, n + 1))
    return_events = []
    noodles_count = [0] * n

    for event in events:
        t_i, w_i, s_i = event

        returned_people = []
        next_return_events = []
        for return_time, person_id in return_events:
            if return_time <= t_i:
                returned_people.append(person_id)
            else:
                next_return_events.append((return_time, person_id))
        return_events = next_return_events
        current_row.extend(returned_people)

        if current_row:
            front_person = current_row.pop(0)
            noodles_count[front_person - 1] += w_i
            return_events.append((t_i + s_i, front_person))
            return_events.sort(key=lambda x: x[0])

    for count in noodles_count:
        print(count)

if __name__ == '__main__':
    solve()