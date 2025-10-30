def solve():
    n, m = map(int, input().split())
    occurrences = []
    for _ in range(m):
        occurrences.append(list(map(int, input().split())))

    noodles_received = [0] * n
    return_times = {}  # person_id -> return_time

    for t, w, s in occurrences:
        current_in_row = set()
        for person in range(1, n + 1):
            if person not in return_times or return_times[person] <= t:
                current_in_row.add(person)

        if current_in_row:
            front_person = min(current_in_row)
            noodles_received[front_person - 1] += w
            return_times[front_person] = t + s

    for amount in noodles_received:
        print(amount)

solve()