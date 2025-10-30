from heapq import heappop, heappush

def solve():
    n = int(input())
    events = [tuple(map(int, input().split())) for _ in range(n)]
    potions = []
    monsters = []
    for t, x in events:
        if t == 1:
            heappush(potions, -x)
        else:
            heappush(monsters, x)
        while monsters and potions and potions[0] < -monsters[0]:
            heappop(potions)
    if monsters:
        print(-1)
    else:
        print(len(potions))
        potions = [-x for x in potions]
        potions.sort()
        actions = [0] * n
        j = 0
        for i in range(n):
            if events[i][0] == 1:
                if events[i][1] == potions[j]:
                    actions[i] = 1
                    j += 1
        print(*actions)

solve()