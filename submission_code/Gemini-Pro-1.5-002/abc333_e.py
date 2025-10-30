# YOUR CODE HERE
def solve():
    n = int(input())
    events = []
    for _ in range(n):
        events.append(list(map(int, input().split())))

    def check(k):
        for start_mask in range(2**sum(1 for t, x in events if t == 1)):
            potions = {}
            max_potions = 0
            current_potions = 0
            actions = []
            mask_index = 0
            defeated = True
            
            for t, x in events:
                if t == 1:
                    take_potion = (start_mask >> mask_index) & 1
                    actions.append(take_potion)
                    mask_index += 1
                    
                    if take_potion:
                        potions[x] = potions.get(x, 0) + 1
                        current_potions += 1
                        max_potions = max(max_potions, current_potions)
                        if max_potions > k:
                            defeated = False
                            break
                else:
                    if potions.get(x, 0) > 0:
                        potions[x] -= 1
                        current_potions -= 1
                    else:
                        defeated = False
                        break
            
            if defeated and max_potions <= k:
                print(k)
                print(*actions)
                return True
        return False

    for k in range(1, n + 1):
        if check(k):
            return

    print(-1)

solve()