def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))
    
    valid_sequences = []
    
    def generate_sequences(index, current_sequence, current_sum):
        if index == n:
            if current_sum % k == 0:
                valid_sequences.append(list(current_sequence))
            return
            
        for v in range(1, r[index] + 1):
            current_sequence.append(v)
            generate_sequences(index + 1, current_sequence, current_sum + v)
            current_sequence.pop() # Backtrack: remove the last added element
            
    generate_sequences(0, [], 0)
    
    for seq in valid_sequences:
        print(*(seq))

if __name__ == '__main__':
    solve()