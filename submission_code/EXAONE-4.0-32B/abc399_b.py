def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    scores = list(map(int, data[1].split()))
    
    indexed_scores = [(i, score) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    ranks = [0] * n
    current_rank = 1
    i = 0
    total = len(indexed_scores)
    while i < total:
        j = i
        current_score = indexed_scores[i][1]
        while j < total and indexed_scores[j][1] == current_score:
            j += 1
        count = j - i
        for k in range(i, j):
            original_index = indexed_scores[k][0]
            ranks[original_index] = current_rank
        current_rank += count
        i = j
        
    for rank in ranks:
        print(rank)

if __name__ == "__main__":
    main()