from typing import List

def insert_sequence(A: List[int]) -> None:
    """
    Insert sequences between adjacent elements in list A as specified in the problem statement.
    """
    i = 0
    while i < len(A) - 1:
        if A[i] < A[i + 1]:
            if A[i + 1] - A[i] > 1:
                for insert_value in range(A[i] + 1, A[i + 1]):
                    A.insert(i + 1, insert_value)
        else:
            if A[i] - A[i + 1] > 1:
                for insert_value in reversed(range(A[i + 1] + 1, A[i])):
                    A.insert(i + 1, insert_value)
        i += 1

def read_input() -> List[int]:
    """
    Read input from stdin and return the list A.
    """
    N = int(input())
    A = list(map(int, input().split()))
    return A

def main() -> None:
    A = read_input()
    insert_sequence(A)
    print(*A)

if __name__ == "__main__":
    main()