def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]
    
    wins = []
    for i in range(N):
        # Count the number of 'o' characters in the i-th string
        w = S[i].count('o')
        wins.append((i+1, w))
    
    # Sort by number of wins descending, then by player number ascending
    wins.sort(key=lambda x: (-x[1], x[0]))
    
    # Print the player numbers in sorted order
    print(' '.join(str(p[0]) for p in wins))

main()