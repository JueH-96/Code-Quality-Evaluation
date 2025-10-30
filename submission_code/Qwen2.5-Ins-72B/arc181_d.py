# YOUR CODE HERE
import sys

def count_inversions(arr):
    inv_count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count

def apply_operations(P, A):
    inv_count = count_inversions(P)
    results = []
    for k in A:
        for i in range(k - 1):
            if P[i] > P[i + 1]:
                P[i], P[i + 1] = P[i + 1], P[i]
                inv_count -= 1
        results.append(inv_count)
    return results

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    M = int(data[N+1])
    A = list(map(int, data[N+2:N+2+M]))
    
    results = apply_operations(P, A)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()