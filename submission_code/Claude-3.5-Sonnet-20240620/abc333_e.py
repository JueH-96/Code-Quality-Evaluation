# YOUR CODE HERE
from collections import defaultdict

def solve():
    N = int(input())
    events = [tuple(map(int, input().split())) for _ in range(N)]
    
    def can_defeat(pickup):
        potions = defaultdict(int)
        for t, x in events:
            if t == 1:
                if pickup[x]:
                    potions[x] += 1
            else:
                if potions[x] == 0:
                    return False
                potions[x] -= 1
        return True
    
    def simulate(pickup):
        max_potions = 0
        potions = defaultdict(int)
        for t, x in events:
            if t == 1 and pickup[x]:
                potions[x] += 1
                max_potions = max(max_potions, sum(potions.values()))
            elif t == 2:
                potions[x] -= 1
        return max_potions
    
    all_potions = [i for i, (t, _) in enumerate(events) if t == 1]
    
    left, right = 0, len(all_potions)
    result = []
    
    while left <= right:
        mid = (left + right) // 2
        pickup = defaultdict(bool)
        
        for i in range(mid):
            pickup[events[all_potions[i]][1]] = True
        
        if can_defeat(pickup):
            result = pickup
            right = mid - 1
        else:
            left = mid + 1
    
    if not result:
        print(-1)
    else:
        k_min = simulate(result)
        print(k_min)
        print(*[int(result[events[i][1]]) for i in all_potions])

solve()