def mex_of_sequence(N, Q, A, queries):
    # To keep track of the presence of numbers in A
    present = set(A)
    
    # Function to find the mex
    def find_mex():
        mex = 0
        while mex in present:
            mex += 1
        return mex

    results = []
    for i, x in queries:
        # Update the value in A
        old_value = A[i - 1]
        A[i - 1] = x
        
        # Update the present set
        present.discard(old_value)
        present.add(x)
        
        # Calculate and store the current mex
        results.append(find_mex())
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    A = list(map(int, data[2:N + 2]))
    queries = []
    
    for j in range(Q):
        i_k = int(data[N + 2 + 2 * j])
        x_k = int(data[N + 2 + 2 * j + 1])
        queries.append((i_k, x_k))
    
    results = mex_of_sequence(N, Q, A, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()