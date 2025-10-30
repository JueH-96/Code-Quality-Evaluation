def solve():
    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        cards.append((a, c, i + 1))

    remaining = list(range(n))
    
    while True:
        discarded = False
        new_remaining = []
        
        for i in remaining:
            can_discard = False
            for j in remaining:
                if i != j:
                    if cards[i][0] < cards[j][0] and cards[i][1] > cards[j][1]:
                        can_discard = True
                        break
            if not can_discard:
                new_remaining.append(i)
            else:
                discarded = True
        
        remaining = new_remaining
        if not discarded:
            break

    result_indices = sorted([cards[i][2] for i in remaining])
    print(len(result_indices))
    print(*result_indices)

solve()