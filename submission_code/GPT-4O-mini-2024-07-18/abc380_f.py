def can_win(takahashi_cards, aoki_cards, table_cards):
    # Sort the cards to make it easier to evaluate moves
    takahashi_cards.sort()
    aoki_cards.sort()
    table_cards.sort()
    
    # Use a tuple to represent the state of the game
    state = (tuple(takahashi_cards), tuple(aoki_cards), tuple(table_cards))
    
    # Memoization to avoid recalculating states
    memo = {}
    
    def dfs(t_cards, a_cards, t_table):
        if (t_cards, a_cards, t_table) in memo:
            return memo[(t_cards, a_cards, t_table)]
        
        # If Takahashi has no cards left, he loses
        if not t_cards:
            return False
        
        # Takahashi's turn
        for i in range(len(t_cards)):
            new_table = list(t_table) + [t_cards[i]]
            new_t_cards = list(t_cards[:i] + t_cards[i+1:])
            
            # Check if he can take any cards from the table
            for card in t_table:
                if card < t_cards[i]:
                    new_t_cards.append(card)
                    new_table.remove(card)
                    break
            
            # Convert back to tuples for memoization
            result = not dfs(tuple(new_t_cards), tuple(a_cards), tuple(new_table))
            if result:
                memo[(t_cards, a_cards, t_table)] = True
                return True
        
        # If no winning move found, Takahashi loses
        memo[(t_cards, a_cards, t_table)] = False
        return False
    
    # Start the game with Takahashi's turn
    if dfs(tuple(takahashi_cards), tuple(aoki_cards), tuple(table_cards)):
        return "Takahashi"
    else:
        return "Aoki"

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, M, L = map(int, data[0].split())
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    C = list(map(int, data[3].split()))
    
    result = can_win(A, B, C)
    print(result)

if __name__ == "__main__":
    main()