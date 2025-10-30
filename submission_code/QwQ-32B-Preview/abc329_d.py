def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:2+M]))
    
    counts = [0] * (N + 1)
    current_max = 0
    winner = 0
    
    for i in range(M):
        candidate = votes[i]
        counts[candidate] += 1
        if counts[candidate] > current_max:
            current_max = counts[candidate]
            winner = candidate
        elif counts[candidate] == current_max:
            if candidate < winner:
                winner = candidate
        print(winner)

if __name__ == "__main__":
    main()