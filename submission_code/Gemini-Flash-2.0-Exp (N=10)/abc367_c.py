def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))
    
    results = []
    
    def generate_sequences(index, current_sequence, current_sum):
        if index == n:
            if current_sum % k == 0:
                results.append(current_sequence.copy())
            return
        
        for i in range(1, r[index] + 1):
            current_sequence.append(i)
            generate_sequences(index + 1, current_sequence, current_sum + i)
            current_sequence.pop()
            
    generate_sequences(0, [], 0)
    
    for seq in results:
        print(*seq)

solve()