from heapq import heappush, heappop

N, T = map(int, input().split())
score_event = []

for _ in range(T):
    A, B = map(int, input().split())
    A -= 1
    heappush(score_event, (B, A))

# simlist is a list of score updates sorted by scores
# The (i, j) tuple represents i points added  to player j, followed by the order in which it was added
simlist = []
distinct = set()
for _ in range(N):
    simlist.append((0, -1))
prev_sim = 0

time = 0.5
# update and print till events remain
while score_event:
    # score_event stores the next updates by time taken to occur
    time += score_event[0][0]
    # remove all events that occur and update simlist
    while score_event and time >= score_event[0][0]:
        ev = heappop(score_event)
        simlist.append((ev[0], ev[1]))
    
    # sort and update simlist
    simlist.sort()
    for i in range(len(simlist)):
        s, p = simlist[i]
        simlist[i] = (s+prev_sim, p)
        distinct.add(s+prev_sim)
    prev_sim = simlist[-1][0]
    print(len(distinct))