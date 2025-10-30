# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = []
    T = []
    for i in range(N-1):
        S.append(int(data[N+1 + 2*i]))
        T.append(int(data[N+2 + 2*i]))
    
    for i in range(N-1):
        max_transfers = A[i] // S[i]
        A[i+1] += max_transfers * T[i]
    
    print(A[-1])

if __name__ == "__main__":
    main()