import sys

def solve(N, events):
    potions = {}
    monsters = {}
    for t, x in events:
        if t == 1:
            if x not in potions:
                potions[x] = 0
            potions[x] += 1
        else:
            if x not in monsters:
                monsters[x] = 0
            monsters[x] += 1
    
    for x in monsters:
        if x not in potions or potions[x] < monsters[x]:
            return -1
    
    K_min = 0
    for x in monsters:
        K_min = max(K_min, potions[x] - monsters[x])
    
    actions = [0] * N
    potion_count = {x: 0 for x in potions}
    for i, (t, x) in enumerate(events):
        if t == 1:
            if potion_count[x] < monsters[x]:
                actions[i] = 1
                potion_count[x] += 1
        else:
            potion_count[x] -= 1
    
    return K_min, actions

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    events = [(int(data[i]), int(data[i+1])) for i in range(1, 2*N, 2)]
    
    result = solve(N, events)
    if result == -1:
        print(-1)
    else:
        K_min, actions = result
        print(K_min)
        print(' '.join(str(a) for i, a in enumerate(actions) if events[i][0] == 1))

if __name__ == "__main__":
    main()