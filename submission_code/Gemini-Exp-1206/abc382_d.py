def solve():
    n, m = map(int, input().split())
    
    result = []
    
    def backtrack(current_sequence):
        if len(current_sequence) == n:
            if current_sequence[-1] <= m:
                result.append(current_sequence.copy())
            return
        
        start = 1 if len(current_sequence) == 0 else current_sequence[-1] + 10
        
        for i in range(start, m + 1):
            
            current_sequence.append(i)
            backtrack(current_sequence)
            current_sequence.pop()

    backtrack([])
    
    filtered_result = []
    for seq in result:
        valid = True
        for i in range(1, len(seq)):
            if seq[i] < seq[i-1] + 10:
                valid = False
                break
        if valid:
            filtered_result.append(seq)
            
    
    print(len(filtered_result))
    for seq in filtered_result:
        print(*seq)

solve()