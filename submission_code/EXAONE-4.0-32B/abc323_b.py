def main():
    n = int(input().strip())
    results = []
    for _ in range(n):
        s = input().strip()
        results.append(s)
    
    wins = [s.count('o') for s in results]
    
    players = []
    for i in range(n):
        players.append((wins[i], i + 1))
        
    sorted_players = sorted(players, key=lambda x: (-x[0], x[1]))
    
    player_numbers = [str(p[1]) for p in sorted_players]
    print(" ".join(player_numbers))

if __name__ == '__main__':
    main()