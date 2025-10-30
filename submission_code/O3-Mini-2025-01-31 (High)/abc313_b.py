def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    M = int(data[1])
    
    # Create a matrix where win[i][j] is True if person i+1 is known to be stronger than person j+1.
    win = [[False for _ in range(N)] for _ in range(N)]
    
    index = 2
    for _ in range(M):
        a = int(data[index]) - 1
        b = int(data[index+1]) - 1
        index += 2
        win[a][b] = True

    # Use transitive closure (Floyd Warshall) to propagate the forced relations.
    for k in range(N):
        for i in range(N):
            if win[i][k]:
                for j in range(N):
                    if win[k][j]:
                        win[i][j] = True

    # Check if there is a candidate i (0-indexed) who is forced to beat every other candidate.
    candidate = -1
    forced_count = 0
    for i in range(N):
        all_stronger = True
        for j in range(N):
            if i == j:
                continue
            if not win[i][j]:
                all_stronger = False
                break
        if all_stronger:
            forced_count += 1
            candidate = i + 1  # converting back to 1-indexed numbering

    # If exactly one candidate is forced to be above all others, output that candidate.
    # Otherwise, output -1.
    if forced_count == 1:
        sys.stdout.write(str(candidate))
    else:
        sys.stdout.write("-1")
        
if __name__ == '__main__':
    main()