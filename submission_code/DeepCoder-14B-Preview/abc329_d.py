def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+M]))
    
    counts = [0] * (N + 1)  # 1-based indexing
    max_votes = 0
    current_min_winner = None
    
    for a in A:
        counts[a] += 1
        if counts[a] > max_votes:
            max_votes = counts[a]
            current_min_winner = a
        elif counts[a] == max_votes:
            if current_min_winner is None or a < current_min_winner:
                current_min_winner = a
        print(current_min_winner)
        
if __name__ == '__main__':
    main()