# YOUR CODE HERE
def process_sequence(n, sequence):
    i = 0
    while i < len(sequence) - 1:
        if abs(sequence[i] - sequence[i + 1]) != 1:
            if sequence[i] < sequence[i + 1]:
                # Insert increasing sequence
                for x in range(sequence[i] + 1, sequence[i + 1]):
                    sequence.insert(i + 1, x)
                    i += 1
            else:
                # Insert decreasing sequence
                for x in range(sequence[i] - 1, sequence[i + 1], -1):
                    sequence.insert(i + 1, x)
                    i += 1
        i += 1
    return sequence

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    sequence = list(map(int, data[1:n+1]))
    
    result = process_sequence(n, sequence)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()