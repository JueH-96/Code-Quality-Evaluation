import math

def is_square(n):
    """Check if a number is a perfect square."""
    if n < 0:
        return False
    sqrt = math.sqrt(n)
    return sqrt == int(sqrt)

def count_pairs(A):
    """Count the number of pairs of integers (i,j) that satisfy both conditions."""
    count = 0
    N = len(A)
    for i in range(N):
        for j in range(i + 1, N):
            if is_square(A[i] * A[j]):
                count += 1
    return count

def main():
    """Read input from stdin and write the answer to stdout."""
    N = int(input())
    A = list(map(int, input().split()))
    answer = count_pairs(A)
    print(answer)

if __name__ == "__main__":
    main()