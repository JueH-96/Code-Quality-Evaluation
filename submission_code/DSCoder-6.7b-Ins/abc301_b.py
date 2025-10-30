def insert_numbers(sequence):
    while True:
        i = 0
        while i < len(sequence) - 1:
            if abs(sequence[i] - sequence[i+1]) != 1:
                break
            i += 1
        else:
            return sequence

        if sequence[i] < sequence[i+1]:
            sequence = sequence[:i+1] + list(range(sequence[i]+1, sequence[i+1])) + sequence[i+1:]
        else:
            sequence = sequence[:i] + list(range(sequence[i]-1, sequence[i+1]-1, -1)) + sequence[i+1:]

N = int(input())
sequence = list(map(int, input().split()))

print(*insert_numbers(sequence))