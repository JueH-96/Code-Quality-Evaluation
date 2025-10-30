def solve():
    a, b, c, d = map(int, input().split())
    cards = [a, b, c, d]
    
    for i in range(1, 14):
        temp_cards = cards + [i]
        counts = {}
        for card in temp_cards:
            if card not in counts:
                counts[card] = 0
            counts[card] += 1
        
        if len(counts) == 2:
            values = list(counts.values())
            if (values[0] == 3 and values[1] == 2) or (values[0] == 2 and values[1] == 3):
                print("Yes")
                return
    
    print("No")

solve()