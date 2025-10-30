import sys

def insert_numbers(sequence):
    i = 0
    while i < len(sequence) - 1:
        if abs(sequence[i] - sequence[i + 1]) != 1:
            if sequence[i] < sequence[i + 1]:
                for num in range(sequence[i] + 1, sequence[i + 1]):
                    sequence.insert(i + 1, num)
                    i += 1
            else:
                for num in range(sequence[i] - 1, sequence[i + 1], -1):
                    sequence.insert(i + 1, num)
                    i += 1
        i += 1
    return sequence

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    result = insert_numbers(A)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()