# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().strip()
    A = list(map(int, input.split()))
    result = sum(A[i] * (2 ** i) for i in range(64))
    print(result)

if __name__ == "__main__":
    main()