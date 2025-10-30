def find_kth_largest(N, K, A, B, C):
    from heapq import nlargest

    # We will use a max-heap to keep track of the K largest values
    max_heap = []

    # Iterate through all combinations of i, j, k
    for i in range(N):
        for j in range(N):
            for k in range(N):
                value = A[i] * B[j] + B[j] * C[k] + C[k] * A[i]
                # We push the value into the heap
                max_heap.append(value)

    # Get the K largest values and return the K-th largest
    kth_largest = nlargest(K, max_heap)[-1]
    return kth_largest

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:2*N+2]))
    C = list(map(int, data[2*N+2:3*N+2]))

    result = find_kth_largest(N, K, A, B, C)
    print(result)

if __name__ == "__main__":
    main()