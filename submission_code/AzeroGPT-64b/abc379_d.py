from heapq import heappush, heappop

Q = int(input())
query = []
events = []
for _ in range(Q):
    q = list(map(int, input().split()))
    query.append(q)

def treat_plant(plant, t):
    return (plant+t, plant)

heaps = [None]
current_time = 0
heappush(events, (plant:=0, t:=0))
harvest_counts = []

for q in query:
    match q:
        case [1]:
            heappush(events, (plant:=0, t:=current_time))
        case [2, t]:
            current_time += t
            events = [treat_plant(p, t) for p, _ in events]
        case [3, th]:
            count = 0
            while events and events[0][0] >= th:
                (_, tp), _ = heappop(events)
                if tp < current_time:
                    count += 1
            harvest_counts.append(count)

print('
'.join(map(str, harvest_counts)))