from heapq import heappush, heappop

N, M = map(int, input().split())
items = [list(map(int, input().split())) for _ in range(N)]

pull_tab_cans = []
regular_cans = []
can_openers = []

for item in items:
    if item[0] == 0:
        pull_tab_cans.append(item[1])
    elif item[0] == 1:
        regular_cans.append(item[1])
    else:
        can_openers.append(item[1])

pull_tab_cans.sort(reverse=True)
regular_cans.sort(reverse=True)
can_openers.sort()

pull_tab_cans = pull_tab_cans[:M]
regular_cans = regular_cans[:M]

happiness = sum(pull_tab_cans)

can_opener_heap = []
for can_opener in can_openers:
    for _ in range(can_opener):
        if regular_cans and (not can_opener_heap or can_opener_heap[0] < regular_cans[-1]):
            heappush(can_opener_heap, regular_cans.pop())
        elif can_opener_heap:
            heappop(can_opener_heap)

while can_opener_heap:
    if regular_cans and (not can_opener_heap or can_opener_heap[0] < regular_cans[-1]):
        happiness += regular_cans.pop()
    elif can_opener_heap:
        happiness += heappop(can_opener_heap)

print(happiness)