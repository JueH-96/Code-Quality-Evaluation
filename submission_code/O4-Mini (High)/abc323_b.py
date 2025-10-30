def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    wins = []
    for _ in range(N):
        s = input().strip()
        # count 'o' for wins
        wins.append(s.count('o'))
    
    # players are 0-indexed internally
    players = list(range(N))
    # sort by wins descending, then by player index ascending
    players.sort(key=lambda i: (-wins[i], i))
    
    # convert to 1-indexed for output
    result = [str(i+1) for i in players]
    print(" ".join(result))

if __name__ == "__main__":
    main()