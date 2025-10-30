import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    events = []
    index = 1
    for i in range(n):
        t = int(data[index])
        x = int(data[index + 1])
        index += 2
        events.append((t, x))
    
    potions = defaultdict(list)
    monsters = defaultdict(list)
    for i, (t, x) in enumerate(events):
        idx = i + 1
        if t == 1:
            potions[x].append(idx)
        else:
            monsters[x].append(idx)
            
    is_matched = [False] * (n + 1)
    intervals = []
    
    types = set(potions.keys()) | set(monsters.keys())
    for x in types:
        P_list = potions[x]
        M_list = monsters[x]
        if len(M_list) > len(P_list):
            print(-1)
            return
            
        P_list.sort(reverse=True)
        M_list.sort(reverse=True)
        
        j = 0
        for m in M_list:
            while j < len(P_list) and P_list[j] >= m:
                j += 1
            if j >= len(P_list):
                print(-1)
                return
            is_matched[P_list[j]] = True
            intervals.append((P_list[j], m))
            j += 1
            
    diff = [0] * (n + 2)
    for start, end in intervals:
        diff[start] += 1
        diff[end] -= 1
        
    current = 0
    max_peak = 0
    for i in range(1, n + 1):
        current += diff[i]
        if current > max_peak:
            max_peak = current
            
    print(max_peak)
    
    actions = []
    for i, (t, x) in enumerate(events):
        idx = i + 1
        if t == 1:
            actions.append('1' if is_matched[idx] else '0')
            
    print(" ".join(actions))

if __name__ == "__main__":
    main()