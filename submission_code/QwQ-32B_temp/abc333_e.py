import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    events = []
    monster_dict = defaultdict(list)
    potion_dict = defaultdict(list)
    for i in range(N):
        t, x = map(int, sys.stdin.readline().split())
        events.append((t, x))
        if t == 1:
            potion_dict[x].append(i + 1)  # event numbers are 1-based
        else:
            monster_dict[x].append(i + 1)
    
    picked = [0] * (N + 2)  # 1-based up to N
    possible = True
    
    for x in monster_dict:
        monsters_x = sorted(monster_dict[x])
        potions_x = sorted(potion_dict.get(x, []))
        if len(potions_x) < len(monsters_x):
            possible = False
            break
        j = len(potions_x) - 1
        assigned = []
        for m in reversed(monsters_x):
            while j >= 0 and potions_x[j] > m:
                j -= 1
            if j < 0:
                possible = False
                break
            assigned.append(potions_x[j])
            j -= 1
        if not possible:
            break
        for p in assigned:
            picked[p] = 1
    
    if not possible:
        print(-1)
        return
    
    current_potions = defaultdict(int)
    total_potions = 0
    max_k = 0
    output_decisions = []
    
    for i in range(N):
        t, x = events[i]
        event_num = i + 1
        if t == 1:
            if picked[event_num]:
                current_potions[x] += 1
                total_potions += 1
                output_decisions.append(1)
            else:
                output_decisions.append(0)
            if total_potions > max_k:
                max_k = total_potions
        else:
            current_potions[x] -= 1
            total_potions -= 1
    
    print(max_k)
    print(' '.join(map(str, output_decisions)))

if __name__ == "__main__":
    main()