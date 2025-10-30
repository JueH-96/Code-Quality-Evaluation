def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    N = int(data[0])
    # Read scores for each person
    scores = list(map(int, data[1:N+1]))

    # The ranking is based on standard competition ranking;
    # A person's rank is 1 + (number of persons with strictly higher score)
    # This matches the given procedure.
    ranks = []
    for score in scores:
        rank = 1 + sum(1 for s in scores if s > score)
        ranks.append(rank)
    
    # Output each person's rank in order (1-indexed persons)
    sys.stdout.write("
".join(map(str, ranks)))

if __name__ == '__main__':
    main()