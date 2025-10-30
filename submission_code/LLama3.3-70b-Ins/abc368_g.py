import sys

def solve(A, B, l, r):
    """Solve the problem for the given range [l, r]"""
    max_val = float('-inf')
    for mask in range(1 << (r - l + 1)):
        val = 0
        for i in range(l, r + 1):
            if (mask & (1 << (i - l))) == 0:
                val += A[i]
            else:
                val *= B[i]
        max_val = max(max_val, val)
    return max_val

def main():
    """Read input and solve the problem"""
    N = int(input())
    A = [0] + list(map(int, input().split()))
    B = [0] + list(map(int, input().split()))
    Q = int(input())
    for _ in range(Q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            A[query[1]] = query[2]
        elif query[0] == 2:
            B[query[1]] = query[2]
        else:
            print(solve(A, B, query[1], query[2]))

if __name__ == "__main__":
    main()