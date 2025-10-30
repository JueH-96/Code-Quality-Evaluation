import sys
import math

def find_A(B):
    for A in range(1, int(math.isqrt(B)) + 2):
        if A ** A == B:
            return A
    return -1

def main():
    input = sys.stdin.read()
    B = int(input.strip())
    result = find_A(B)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()