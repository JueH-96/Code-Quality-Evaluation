import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    freq = defaultdict(int)
    present = set()

    for num in A:
        freq[num] += 1
        present.add(num)

    for _ in range(Q):
        i = int(input[ptr]) - 1
        ptr += 1
        x = int(input[ptr])
        ptr += 1

        old_val = A[i]

        # Remove old_val
        freq[old_val] -= 1
        if freq[old_val] == 0:
            present.discard(old_val)

        # Add x
        freq[x] += 1
        if freq[x] == 1:
            present.add(x)
        A[i] = x

        # Compute mex
        mex = 0
        while mex in present:
            mex += 1
        print(mex)

if __name__ == '__main__':
    main()