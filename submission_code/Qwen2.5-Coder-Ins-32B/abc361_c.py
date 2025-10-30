# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:]))

    A.sort()
    min_diff = float('inf')
    
    for i in range(N - K + 1):
        min_diff = min(min_diff, A[i + K - 1] - A[i])
    
    print(min_diff)

if __name__ == "__main__":
    main()