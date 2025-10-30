import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    sequence = deque()

    for i in range(N):
        sequence.append(A[i])
        while len(sequence) > 1 and sequence[-1] == sequence[-2]:
            sequence.pop()
            sequence.pop()
            sequence.append(A[i] + 1)

    print(len(sequence))

if __name__ == "__main__":
    main()