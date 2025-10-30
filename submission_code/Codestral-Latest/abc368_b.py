import sys
from typing import List

def count_operations(A: List[int]) -> int:
    count = 0
    while True:
        # Sort A in descending order
        A.sort(reverse=True)

        # Check if there is at most one positive element
        if A[1] == 0:
            break

        # Decrease both A_1 and A_2 by 1
        A[0] -= 1
        A[1] -= 1
        count += 1

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = count_operations(A)
    print(result)

if __name__ == "__main__":
    main()