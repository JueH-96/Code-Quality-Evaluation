import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    all_events = []
    for _ in range(N):
        t_i, x_i = map(int, sys.stdin.readline().split())
        all_events.append((t_i, x_i))
    
    x_dict = defaultdict(list)
    for idx in range(N):
        t_i, x_i = all_events[idx]
        x_dict[x_i].append((idx, t_i, x_i))
    
    picked = [0] * N
    events_for_k = []

    for x in x_dict:
        events = x_dict[x]
        potions = []
        monsters = []
        for (idx, t, x_val) in events:
            if t == 1:
                potions.append(idx)
            else:
                monsters.append(idx)
        
        if len(potions) < len(monsters):
            print(-1)
            return
        
        monsters_reversed = monsters[::-1]
        assigned_potions = []
        j = len(potions) - 1
        for m in monsters_reversed:
            while j >= 0 and potions[j] > m:
                j -= 1
            if j < 0:
                print(-1)
                return
            assigned_potions.append(potions[j])
            j -= 1
        
        for p in assigned_potions:
            picked[p] = 1
        
        for i in range(len(assigned_potions)):
            p_gi = assigned_potions[i]
            m_gi = monsters_reversed[i]
            events_for_k.append((p_gi, 1))
            events_for_k.append((m_gi, -1))
    
    events_for_k.sort(key=lambda x: (x[0], -x[1]))
    current = 0
    max_k = 0
    for event in events_for_k:
        current += event[1]
        if current > max_k:
            max_k = current
    
    print(max_k)
    output = []
    for idx in range(N):
        if all_events[idx][0] == 1:
            output.append('1' if picked[idx] else '0')
    print(' '.join(output))

if __name__ == "__main__":
    main()