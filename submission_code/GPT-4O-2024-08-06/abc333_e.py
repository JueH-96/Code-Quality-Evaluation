# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    events = []
    for i in range(1, len(data), 2):
        t = int(data[i])
        x = int(data[i+1])
        events.append((t, x))
    
    potions = {}
    actions = []
    max_potions = 0
    
    for t, x in events:
        if t == 1:
            # Find a potion of type x
            if x not in potions:
                potions[x] = 0
            potions[x] += 1
            actions.append(1)
            max_potions = max(max_potions, sum(potions.values()))
        elif t == 2:
            # Encounter a monster of type x
            if x in potions and potions[x] > 0:
                potions[x] -= 1
            else:
                print(-1)
                return
    
    # Calculate K_min
    potions = {}
    actions = []
    current_potions = 0
    K_min = max_potions
    
    for t, x in events:
        if t == 1:
            # Find a potion of type x
            if x not in potions:
                potions[x] = 0
            if current_potions < K_min:
                potions[x] += 1
                current_potions += 1
                actions.append(1)
            else:
                actions.append(0)
        elif t == 2:
            # Encounter a monster of type x
            if x in potions and potions[x] > 0:
                potions[x] -= 1
                current_potions -= 1
    
    print(K_min)
    print(" ".join(map(str, actions)))

main()