def count_inversions(arr):
    n = len(arr)
    inv_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count

def main():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # For each k from 0 to M-1
    for k in range(M):
        # Create sequence B where each element is (A[i] + k) % M
        B = [(a + k) % M for a in A]
        # Calculate and print inversion number for this sequence
        result = count_inversions(B)
        print(result)

if __name__ == "__main__":
    main()