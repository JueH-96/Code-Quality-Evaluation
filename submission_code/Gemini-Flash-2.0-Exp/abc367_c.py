def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))
    
    sequences = []
    
    def find_sequences(index, current_sequence, current_sum):
        if index == n:
            if current_sum % k == 0:
                sequences.append(tuple(current_sequence))
            return
        
        for i in range(1, r[index] + 1):
            current_sequence.append(i)
            find_sequences(index + 1, current_sequence, current_sum + i)
            current_sequence.pop()
            
    find_sequences(0, [], 0)
    
    sequences.sort()
    
    for sequence in sequences:
        print(*sequence)

solve()