def solve():
    n, m = map(int, input().split())
    valid_sequences = []

    def find_sequences(index, current_sequence):
        if index == n:
            valid_sequences.append(tuple(current_sequence))
            return

        if index == 0:
            upper_bound = m - (n - 1) * 10
            for val in range(1, upper_bound + 1):
                find_sequences(index + 1, [val])
        else:
            min_val = current_sequence[-1] + 10
            upper_bound = m - (n - 1 - index) * 10
            for val in range(min_val, upper_bound + 1):
                find_sequences(index + 1, current_sequence + [val])

    find_sequences(0, [])
    print(len(valid_sequences))
    for seq in valid_sequences:
        print(*seq)

solve()