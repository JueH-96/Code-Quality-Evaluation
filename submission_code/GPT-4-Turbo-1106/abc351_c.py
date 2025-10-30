from collections import deque

def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))

    sequence = deque()
    for a in A:
        sequence.append(a)
        while len(sequence) > 1 and sequence[-1] == sequence[-2]:
            sequence.pop()
            sequence[-1] += 1

    print(len(sequence))

if __name__ == "__main__":
    main()