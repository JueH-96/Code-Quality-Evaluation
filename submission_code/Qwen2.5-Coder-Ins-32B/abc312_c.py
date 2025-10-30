# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+N]))
    B = list(map(int, input[2+N:2+N+M]))
    
    A.sort()
    B.sort(reverse=True)
    
    i, j = 0, 0
    while i < N and j < M:
        if A[i] <= B[j]:
            return A[i]
        i += 1
        j += 1
    
    if i < N:
        return A[i]
    else:
        return B[j-1] + 1

if __name__ == "__main__":
    print(main())