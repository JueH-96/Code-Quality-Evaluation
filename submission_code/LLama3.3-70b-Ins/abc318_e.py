import sys

def count_triples(N, A):
    """
    Count the number of triples (i, j, k) that satisfy the conditions:
    1 <= i < j < k <= N, A_i = A_k, and A_i != A_j.
    
    Args:
    N (int): The length of the sequence A.
    A (list): A sequence of positive integers.
    
    Returns:
    int: The number of triples that satisfy the conditions.
    """
    count = 0
    for i in range(N - 2):  # i < j < k, so i can be at most N - 2
        for j in range(i + 1, N - 1):  # j < k, so j can be at most N - 1
            for k in range(j + 1, N):  # k can be at most N
                if A[i] == A[k] and A[i] != A[j]:
                    count += 1
    return count

def main():
    # Read input from stdin
    N = int(input())
    A = list(map(int, input().split()))
    
    # Count the number of triples
    count = count_triples(N, A)
    
    # Print the result to stdout
    print(count)

if __name__ == "__main__":
    main()