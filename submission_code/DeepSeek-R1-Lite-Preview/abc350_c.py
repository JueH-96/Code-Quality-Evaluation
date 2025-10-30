import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize position list with 1-based indexing
    pos = [0] * (N + 1)
    for idx, val in enumerate(A):
        pos[val] = idx + 1  # positions are 1-based
    
    swaps = []
    for i in range(1, N + 1):
        if A[i - 1] != i:
            j = pos[i]
            swaps.append((i, j))
            # Swap elements at positions i-1 and j-1
            A[i - 1], A[j - 1] = A[j - 1], A[i - 1]
            # Update positions after swap
            pos[A[i - 1]] = i
            pos[A[j - 1]] = j
    
    print(len(swaps))
    for swap in swaps:
        print(swap[0], swap[1])

if __name__ == '__main__':
    main()