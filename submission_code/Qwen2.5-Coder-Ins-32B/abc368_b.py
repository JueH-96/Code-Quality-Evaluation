# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    
    operations = 0
    while sum(x > 0 for x in A) > 1:
        A.sort(reverse=True)
        A[0] -= 1
        A[1] -= 1
        operations += 1
    
    print(operations)

if __name__ == "__main__":
    main()