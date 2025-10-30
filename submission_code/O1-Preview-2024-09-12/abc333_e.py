# YOUR CODE HERE
N = int(input())
events = []
types = set()
for _ in range(N):
    t_i, x_i = map(int, input().split())
    events.append((t_i, x_i))
    types.add(x_i)

pick = [0]*N  # pick[i]=1 if we pick up the potion at event i

needs = {}
for x in types:
    needs[x] = 0

for i in range(N-1, -1, -1):
    t_i, x_i = events[i]
    if t_i == 2:  # Monster
        needs[x_i] +=1
    elif t_i ==1:  # Potion
        if needs[x_i]>0:
            pick[i]=1
            needs[x_i]-=1

impossible = False
for x in needs:
    if needs[x]>0:
        print(-1)
        impossible = True
        break

if not impossible:
    counts = {}
    for x in types:
        counts[x]=0
    total_potions = 0
    K = 0
    pick_output = []
    potion_event_indices = []
    potion_event_count = 0
    for i in range(N):
        t_i, x_i = events[i]
        if t_i ==1:  # Potion
            if pick[i]==1:
                counts[x_i]=counts.get(x_i, 0)+1
                total_potions +=1
            K = max(K, total_potions)
            if pick[i]==1:
                pick_output.append('1')
            else:
                pick_output.append('0')
        elif t_i ==2:
            if counts.get(x_i,0)>0:
                counts[x_i]-=1
                total_potions -=1
            else:
                print(-1)
                impossible = True
                break

    if not impossible:
        print(K)
        print(' '.join(pick_output))