import heapq
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    events = []
    potions_by_type = [[] for _ in range(n+1)]
    monsters_by_type = [[] for _ in range(n+1)]
    
    for i in range(1, n+1):
        line = data[i].split()
        if not line:
            continue
        t = int(line[0])
        x = int(line[1])
        events.append((t, x))
        if t == 1:
            if x <= n:
                potions_by_type[x].append(i-1)
        else:
            if x <= n:
                monsters_by_type[x].append(i-1)
                
    feasible = True
    for x in range(1, n+1):
        monsters = monsters_by_type[x]
        potions = potions_by_type[x]
        if not monsters:
            continue
        if len(potions) < len(monsters):
            feasible = False
            break
        monsters_sorted = sorted(monsters)
        potions_sorted = sorted(potions)
        for j in range(len(monsters)):
            if potions_sorted[j] > monsters_sorted[j]:
                feasible = False
                break
    if not feasible:
        print(-1)
        return
        
    chosen_events = set()
    for x in range(1, n+1):
        monsters = monsters_by_type[x]
        potions = potions_by_type[x]
        if not monsters:
            continue
        monsters_sorted = sorted(monsters)
        potions_sorted = sorted(potions)
        heap = []
        j = 0
        for m in monsters_sorted:
            while j < len(potions_sorted) and potions_sorted[j] <= m:
                heapq.heappush(heap, -potions_sorted[j])
                j += 1
            if not heap:
                feasible = False
                break
            else:
                selected_potion = -heapq.heappop(heap)
                chosen_events.add(selected_potion)
        if not feasible:
            break
            
    if not feasible:
        print(-1)
        return
        
    current_count = 0
    max_val = 0
    output_actions = []
    for i in range(n):
        t, x = events[i]
        if t == 1:
            if i in chosen_events:
                current_count += 1
                output_actions.append('1')
            else:
                output_actions.append('0')
        else:
            current_count -= 1
        if current_count > max_val:
            max_val = current_count
            
    print(max_val)
    print(" ".join(output_actions))

if __name__ == "__main__":
    main()