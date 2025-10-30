# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    events = []
    for _ in range(m):
        events.append(list(map(int, input().split())))

    noodles = [0] * n
    queue = list(range(n))
    return_times = [0] * n

    for event in events:
        t, w, s = event
        
        while queue and return_times[queue[0]] <= t:
            queue.pop(0)

        if queue:
            person_index = queue.pop(0)
            noodles[person_index] += w
            return_times[person_index] = t + s
            queue.append(person_index)
            queue.sort(key=lambda x: return_times[x])


    for noodle_count in noodles:
        print(noodle_count)

solve()