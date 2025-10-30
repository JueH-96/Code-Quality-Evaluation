def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:2+M]))
    
    counts = [0] * (N + 1)
    current_winner = None
    results = []
    
    for vote in votes:
        counts[vote] += 1
        if current_winner is None:
            current_winner = vote
        else:
            if counts[vote] > counts[current_winner]:
                current_winner = vote
            elif counts[vote] == counts[current_winner] and vote < current_winner:
                current_winner = vote
        results.append(str(current_winner))
    
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()