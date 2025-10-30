import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    A = list(map(int, data[idx:idx+N]))
    idx += N
    queries = []
    for _ in range(Q):
        i = int(data[idx]) - 1
        x = int(data[idx+1])
        queries.append((i, x))
        idx += 2
    
    # Initialize frequency map
    freq = defaultdict(int)
    for num in A:
        if num <= 2 * 10**5:
            freq[num] += 1
    
    # Initialize mex
    mex = 0
    while freq[mex] > 0:
        mex += 1
    
    for i, x in queries:
        old = A[i]
        if old <= 2 * 10**5:
            freq[old] -= 1
            if freq[old] == 0 and old < mex:
                mex = old
        A[i] = x
        if x <= 2 * 10**5:
            freq[x] += 1
            if x == mex:
                while freq[mex] > 0:
                    mex += 1
        print(mex)

if __name__ == "__main__":
    main()