# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    swaps = []
    
    # We need to sort A to (1, 2, ..., N)
    for i in range(N):
        while A[i] != i + 1:
            # Swap A[i] with A[A[i] - 1]
            correct_pos = A[i] - 1
            A[i], A[correct_pos] = A[correct_pos], A[i]
            swaps.append((i + 1, correct_pos + 1))
    
    # Output the result
    print(len(swaps))
    for i, j in swaps:
        print(i, j)