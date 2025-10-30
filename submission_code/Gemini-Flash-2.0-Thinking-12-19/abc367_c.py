def solve():
    n, k = map(int, input().split())
    r = list(map(int, input().split()))
    
    def generate_sequences(index, current_sequence, current_sum):
        if index == n:
            if current_sum % k == 0:
                print(*(current_sequence))
            return
        
        for value in range(1, r[index] + 1):
            next_sequence = list(current_sequence)
            next_sequence.append(value)
            generate_sequences(index + 1, next_sequence, current_sum + value)
            
    generate_sequences(0, [], 0)

if __name__ == '__main__':
    solve()