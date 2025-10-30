import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Create pos list with 1-based indexing
    pos = [0] * (N + 1)
    for idx, val in enumerate(A):
        pos[val] = idx + 1  # 1-based index
    
    swaps = []
    for i in range(1, N+1):
        if A[i-1] != i:
            j = pos[i]
            swaps.append((i, j))
            # Swap A[i-1] and A[j-1]
            A[i-1], A[j-1] = A[j-1], A[i-1]
            # Update pos
            pos[A[i-1]] = i
            pos[A[j-1]] = j
    
    K = len(swaps)
    print(K)
    for swap in swaps:
        print(swap[0], swap[1])

if __name__ == "__main__":
    main()