# YOUR CODE HERE
import sys
import sys
import sys
from collections import defaultdict

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx]); idx +=1
    Q = int(data[idx]); idx +=1
    A = list(map(int, data[idx:idx+N]))
    idx +=N
    queries = []
    for _ in range(Q):
        i = int(data[idx])-1
        x = int(data[idx+1])
        queries.append((i,x))
        idx +=2

    freq = defaultdict(int)
    for num in A:
        freq[num] +=1

    current_mex =0
    while freq[current_mex] >0:
        current_mex +=1

    for i,x in queries:
        y = A[i]
        freq[y] -=1
        if y < current_mex and freq[y] ==0:
            current_mex = y
        A[i] = x
        freq[x] +=1
        if x == current_mex:
            while freq[current_mex] >0:
                current_mex +=1
        print(current_mex)

if __name__ == "__main__":
    main()