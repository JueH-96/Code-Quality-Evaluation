import sys

def solve():
    n = int(input())
    events = []
    for _ in range(n):
        t, x = map(int, input().split())
        events.append((t, x))

    def can_defeat(monsters, potions):
        for monster in monsters:
            if monster not in potions:
                return False
            potions.remove(monster)
        return True

    def get_min_k(events):
        monsters = [x for t, x in events if t == 2]
        potions = []
        max_potions = 0
        min_k = float('inf')
        for t, x in events:
            if t == 1:
                potions.append(x)
                max_potions = max(max_potions, len(potions))
            else:
                if x not in potions:
                    return -1
                potions.remove(x)
        for k in range(max_potions, 0, -1):
            for mask in range(1 << len([t for t, x in events if t == 1])):
                current_potions = []
                current_max_potions = 0
                for i, (t, x) in enumerate(events):
                    if t == 1 and (mask & (1 << i)) != 0:
                        current_potions.append(x)
                        current_max_potions = max(current_max_potions, len(current_potions))
                    elif t == 2:
                        if x not in current_potions:
                            break
                        current_potions.remove(x)
                else:
                    if current_max_potions <= k and can_defeat(monsters, current_potions[:]):
                        min_k = min(min_k, current_max_potions)
                        if min_k == k:
                            return min_k, [int((mask & (1 << i)) != 0) for i, (t, x) in enumerate(events) if t == 1]
        return -1, []

    min_k, actions = get_min_k(events)
    if min_k == -1:
        print(-1)
    else:
        print(min_k)
        print(' '.join(map(str, actions)))

if __name__ == '__main__':
    solve()