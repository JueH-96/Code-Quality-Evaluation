# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    empty_seats = K
    starts = 0
    
    for i in range(N):
        if A[i] > empty_seats:
            starts += 1
            empty_seats = K - A[i]
        else:
            empty_seats -= A[i]
    
    if empty_seats < K:
        starts += 1
    
    print(starts)

if __name__ == "__main__":
    main()