import sys
input = sys.stdin.read
from collections import defaultdict

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    A = list(map(int, data[index:index+N]))
    index += N
    
    queries = []
    for _ in range(Q):
        i_k = int(data[index]) - 1
        index += 1
        x_k = int(data[index])
        index += 1
        queries.append((i_k, x_k))
    
    # To find mex efficiently, we need to track the counts of each number
    counts = defaultdict(int)
    for num in A:
        counts[num] += 1
    
    # To find the smallest missing number efficiently
    mex = 0
    while counts[mex] > 0:
        mex += 1
    
    results = []
    
    for i_k, x_k in queries:
        old_value = A[i_k]
        A[i_k] = x_k
        
        if old_value != x_k:
            # Decrease the count of the old value
            counts[old_value] -= 1
            # If the count of the old value becomes zero and it's less than current mex
            if counts[old_value] == 0 and old_value < mex:
                mex = old_value
            
            # Increase the count of the new value
            counts[x_k] += 1
            # If the new value is exactly mex and now exists, find the next mex
            if x_k == mex:
                while counts[mex] > 0:
                    mex += 1
        
        results.append(str(mex))
    
    print("
".join(results))

if __name__ == "__main__":
    main()