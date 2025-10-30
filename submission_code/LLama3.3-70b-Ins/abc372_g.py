import sys

def solve(N, A, B, C):
    """
    Find the number of pairs of positive integers (x, y) that satisfy the condition:
    A_i * x + B_i * y < C_i for all 1 <= i <= N.

    Args:
    N (int): The number of sequences.
    A (list): The first sequence of positive integers.
    B (list): The second sequence of positive integers.
    C (list): The third sequence of positive integers.

    Returns:
    int: The number of pairs of positive integers (x, y) that satisfy the condition.
    """
    max_x = min((C[i] - 1) // A[i] for i in range(N))
    max_y = min((C[i] - 1) // B[i] for i in range(N))

    count = 0
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            if all(A[i] * x + B[i] * y < C[i] for i in range(N)):
                count += 1

    return count

def main():
    T = int(input())
    for _ in range(T):
        N = int(input())
        A, B, C = [], [], []
        for _ in range(N):
            a, b, c = map(int, input().split())
            A.append(a)
            B.append(b)
            C.append(c)

        result = solve(N, A, B, C)
        print(result)

if __name__ == "__main__":
    main()