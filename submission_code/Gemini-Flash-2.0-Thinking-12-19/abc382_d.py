def solve():
    n, m = map(int, input().split())
    valid_sequences = []
    
    def find_sequences(index, current_prefix):
        if index > n:
            valid_sequences.append(tuple(current_prefix))
            return
        if index == 1:
            upper_bound = m - (n - 1) * 10
            for v in range(1, upper_bound + 1):
                if v <= upper_bound:
                    find_sequences(index + 1, current_prefix + [v])
        else:
            start_value = current_prefix[-1] + 10
            if index < n:
                upper_bound = m - (n - index) * 10
            else:
                upper_bound = m
            for v in range(start_value, upper_bound + 1):
                if v <= upper_bound:
                    find_sequences(index + 1, current_prefix + [v])
                    
    find_sequences(1, [])
    
    print(len(valid_sequences))
    for seq in valid_sequences:
        print(*(list(seq)))

if __name__ == '__main__':
    solve()