def insert_numbers(sequence):
    n = len(sequence)
    i = 0
    while i < n - 1:
        if abs(sequence[i] - sequence[i + 1]) != 1:
            if sequence[i] < sequence[i + 1]:
                sequence = sequence[:i + 1] + list(range(sequence[i] + 1, sequence[i + 1])) + sequence[i + 1:]
            else:
                sequence = sequence[:i + 1] + list(range(sequence[i] - 1, sequence[i + 1], -1)) + sequence[i + 1:]
            n = len(sequence)
        else:
            i += 1
    return sequence

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = insert_numbers(A)
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()