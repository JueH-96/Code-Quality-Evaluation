# YOUR CODE HERE
import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    while True:
        found = False
        for i in range(len(A)-1):
            diff = abs(A[i] - A[i+1])
            if diff !=1:
                found = True
                if A[i] < A[i+1]:
                    insert = list(range(A[i]+1, A[i+1]))
                else:
                    insert = list(range(A[i]-1, A[i+1], -1))
                A = A[:i+1] + insert + A[i+1:]
                break
        if not found:
            break
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()