def solve():
    n, m = map(int, input().split())
    events = []
    for _ in range(m):
        events.append(list(map(int, input().split())))

    noodles = [0] * n
    queue = list(range(n))
    out_time = {}

    for t, w, s in events:
        
        to_remove = []
        for p in out_time:
            if out_time[p] <= t:
                
                insert_pos = 0
                for i in range(len(queue)):
                    if queue[i] > p:
                        break
                    insert_pos += 1
                queue.insert(insert_pos, p)
                to_remove.append(p)
        for p in to_remove:
            del out_time[p]

        if queue:
            person = queue.pop(0)
            noodles[person] += w
            out_time[person] = t + s

    for amount in noodles:
        print(amount)

solve()