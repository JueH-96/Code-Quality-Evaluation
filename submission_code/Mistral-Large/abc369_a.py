import sys

def count_arithmetic_sequences(A, B):
    count = 0
    for x in range(min(A, B) - 2 * (max(A, B) - min(A, B)), max(A, B) + 2 * (max(A, B) - min(A, B)) + 1):
        if (A + x) // 2 == B or (B + x) // 2 == A or (A + B) // 2 == x:
            count += 1
    return count

def main():
    input = sys.stdin.read()
    A, B = map(int, input.split())
    result = count_arithmetic_sequences(A, B)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()