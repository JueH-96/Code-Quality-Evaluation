def solve():
    n, m = map(int, input().split())
    
    ans = []
    
    def find_sequences(current_sequence):
        if len(current_sequence) == n:
            if current_sequence[-1] <= m:
                ans.append(current_sequence.copy())
            return
        
        start = 1 if len(current_sequence) == 0 else current_sequence[-1] + 10
        end = m
        if len(current_sequence) > 0:
            remaining_elements = n - len(current_sequence) -1
            end = min(m, m - remaining_elements * 10)
        
        for i in range(start, end + 1):
            current_sequence.append(i)
            find_sequences(current_sequence)
            current_sequence.pop()

    find_sequences([])
    print(len(ans))
    for seq in ans:
        print(*seq)

solve()