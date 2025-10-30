def main():
    n = int(input().strip())
    scores = list(map(int, input().split()))
    arr = [(scores[i], i) for i in range(n)]
    arr.sort(key=lambda x: x[0], reverse=True)
    
    current_rank = 1
    result = [0] * n
    i = 0
    while i < n:
        j = i
        current_score = arr[i][0]
        while j < n and arr[j][0] == current_score:
            j += 1
        count = j - i
        for idx in range(i, j):
            original_index = arr[idx][1]
            result[original_index] = current_rank
        current_rank += count
        i = j
        
    for rank in result:
        print(rank)

if __name__ == '__main__':
    main()