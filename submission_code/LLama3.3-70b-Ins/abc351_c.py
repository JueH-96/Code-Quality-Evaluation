import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    sequence = []
    for a in A:
        sequence.append(2**a)
        while len(sequence) > 1 and sequence[-1] == sequence[-2]:
            sequence.pop()
            sequence.pop()
            sequence.append(2**(a+1))
    print(len(sequence))

if __name__ == "__main__":
    solve()