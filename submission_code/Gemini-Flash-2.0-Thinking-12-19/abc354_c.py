import sys

def solve():
    n = int(sys.stdin.readline())
    cards = []
    for i in range(n):
        a, c = map(int, sys.stdin.readline().split())
        cards.append({'strength': a, 'cost': c, 'original_index': i + 1})
    
    sorted_cards = sorted(cards, key=lambda card: card['strength'], reverse=True)
    
    min_cost = float('inf')
    remaining_indices = []
    
    for card in sorted_cards:
        if card['cost'] < min_cost:
            min_cost = card['cost']
            remaining_indices.append(card['original_index'])
            
    remaining_indices.sort()
    
    print(len(remaining_indices))
    print(*(remaining_indices))

if __name__ == '__main__':
    solve()