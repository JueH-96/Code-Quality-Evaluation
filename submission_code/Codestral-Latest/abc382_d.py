def generate_sequences(N, M):
    sequences = []

    def backtrack(index, current_sequence):
        if index == N:
            if current_sequence[-1] <= M:
                sequences.append(tuple(current_sequence))
            return

        start = current_sequence[-1] + 10 if index > 0 else 1
        for i in range(start, M + 1):
            current_sequence.append(i)
            backtrack(index + 1, current_sequence)
            current_sequence.pop()

    backtrack(0, [])
    return sequences

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])

    sequences = generate_sequences(N, M)
    sequences.sort()

    print(len(sequences))
    for seq in sequences:
        print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()