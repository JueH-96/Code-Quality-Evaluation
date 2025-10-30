import heapq

n, m = map(int, input().split())

noodles = [0] * (n + 1)
in_row_heap = list(range(1, n + 1))
heapq.heapify(in_row_heap)
return_heap = []
removed = set()

for _ in range(m):
    t, w, s = map(int, input().split())
    
    # Process returns
    while return_heap and return_heap[0][0] <= t:
        return_time, person_id = heapq.heappop(return_heap)
        heapq.heappush(in_row_heap, person_id)
        removed.discard(person_id)
    
    # Find front person
    while in_row_heap and in_row_heap[0] in removed:
        heapq.heappop(in_row_heap)
    
    if in_row_heap:
        front_person = heapq.heappop(in_row_heap)
        noodles[front_person] += w
        removed.add(front_person)
        heapq.heappush(return_heap, (t + s, front_person))

for i in range(1, n + 1):
    print(noodles[i])