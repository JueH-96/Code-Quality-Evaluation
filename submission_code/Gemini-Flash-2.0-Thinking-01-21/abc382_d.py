def solve():
    n, m = map(int, input().split())
    solutions = []
    
    def get_sequences(index, current_seq):
        if index > n:
            solutions.append(tuple(current_seq))
            return
        
        if index == 1:
            start_val = 1
        else:
            start_val = current_seq[-1] + 10
        
        end_val = m - 10 * (n - index)
        
        if start_val <= end_val:
            for val in range(start_val, end_val + 1):
                get_sequences(index + 1, current_seq + [val])
            
    get_sequences(1, [])
    print(len(solutions))
    for seq in solutions:
        print(*(list(seq)))

if __name__ == '__main__':
    solve()