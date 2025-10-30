def solve():
    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        cards.append((a, c, i + 1))

    remaining = list(range(n))
    
    changed = True
    while changed:
        changed = False
        to_remove = set()
        for i in range(len(remaining)):
            for j in range(len(remaining)):
                if i != j:
                    x = remaining[i]
                    y = remaining[j]
                    if cards[x][0] > cards[y][0] and cards[x][1] < cards[y][1]:
                        to_remove.add(y)
                        changed = True
        
        new_remaining = []
        for i in remaining:
            if i not in to_remove:
                new_remaining.append(i)
        remaining = new_remaining

    result = sorted([cards[i][2] for i in remaining])
    print(len(result))
    print(*result)

solve()