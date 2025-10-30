def solve():
    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        cards.append({'strength': a, 'cost': c, 'original_index': i + 1})
    
    sorted_cards = sorted(cards, key=lambda card: card['strength'], reverse=True)
    kept_cards = []
    
    for current_card in sorted_cards:
        discard = False
        for kept_card in kept_cards:
            if kept_card['strength'] > current_card['strength'] and kept_card['cost'] < current_card['cost']:
                discard = True
                break
        if not discard:
            kept_cards.append(current_card)
            
    remaining_indices = sorted([card['original_index'] for card in kept_cards])
    
    print(len(remaining_indices))
    print(*(remaining_indices))

if __name__ == '__main__':
    solve()