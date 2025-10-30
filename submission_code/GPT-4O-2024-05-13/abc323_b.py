# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1:N+1]
    
    wins = []
    
    for i in range(N):
        win_count = S[i].count('o')
        wins.append((win_count, i + 1))
    
    wins.sort(key=lambda x: (-x[0], x[1]))
    
    result = [str(player[1]) for player in wins]
    print(" ".join(result))

if __name__ == "__main__":
    main()