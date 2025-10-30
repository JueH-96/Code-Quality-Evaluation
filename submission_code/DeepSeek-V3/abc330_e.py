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
        i = int(data[idx])
        x = int(data[idx+1])
        queries.append((i-1, x))  # converting to 0-based index
        idx += 2
    
    # Initialize frequency map
    freq = defaultdict(int)
    for num in A:
        if num <= 2 * 10**5:
            freq[num] += 1
    
    # Initialize mex
    mex = 0
    while freq.get(mex, 0) > 0:
        mex += 1
    
    for i, x in queries:
        old_val = A[i]
        if old_val <= 2 * 10**5:
            freq[old_val] -= 1
            if freq[old_val] == 0 and old_val < mex:
                mex = min(mex, old_val)
        A[i] = x
        if x <= 2 * 10**5:
            freq[x] += 1
            if x == mex:
                while freq.get(mex, 0) > 0:
                    mex += 1
        print(mex)

if __name__ == "__main__":
    main()