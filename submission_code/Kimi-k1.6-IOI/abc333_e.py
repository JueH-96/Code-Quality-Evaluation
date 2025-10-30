import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    events = []
    potions = defaultdict(list)
    monsters = defaultdict(list)
    idx = 1
    for i in range(n):
        t = int(input[idx])
        x = int(input[idx+1])
        events.append((t, x))
        if t == 1:
            potions[x].append(i + 1)  # 1-based index
        else:
            monsters[x].append(i + 1)
        idx += 2
    
    assigned = set()
    possible = True
    
    for x in monsters:
        ms = monsters[x]
        ps = potions[x]
        if len(ps) < len(ms):
            possible = False
            break
        j = len(ps) - 1
        for m in reversed(ms):
            while j >= 0 and ps[j] >= m:
                j -= 1
            if j >= 0:
                assigned.add(ps[j])
                j -= 1
            else:
                possible = False
                break
        if not possible:
            break
    
    if not possible:
        print(-1)
        return
    
    choices = []
    for i in range(n):
        t, x = events[i]
        if t == 1:
            event_idx = i + 1
            choices.append(1 if event_idx in assigned else 0)
    
    max_k = 0
    current = 0
    for i in range(n):
        t, x = events[i]
        if t == 1:
            event_idx = i + 1
            if event_idx in assigned:
                current += 1
                if current > max_k:
                    max_k = current
        else:
            current -= 1
    
    print(max_k)
    print(' '.join(map(str, choices)))

if __name__ == "__main__":
    main()