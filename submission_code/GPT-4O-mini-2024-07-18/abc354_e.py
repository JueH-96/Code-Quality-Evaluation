def can_remove_pair(cards, used):
    n = len(cards)
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if used[j]:
                continue
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                return True
    return False

def game_winner(cards, used):
    if not can_remove_pair(cards, used):
        return 0  # No moves left, current player loses

    current_player_wins = False
    n = len(cards)
    
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if used[j]:
                continue
            if cards[i][0] == cards[j][0] or cards[i][1] == cards[j][1]:
                # Try removing these two cards
                used[i] = True
                used[j] = True
                
                # Check if the opponent loses
                if game_winner(cards, used) == 0:
                    current_player_wins = True
                
                # Backtrack
                used[i] = False
                used[j] = False

    return 1 if current_player_wins else 0

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    cards = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    used = [False] * N
    result = game_winner(cards, used)
    
    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()