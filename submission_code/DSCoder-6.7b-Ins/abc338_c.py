import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    Q = list(map(int, sys.stdin.readline().strip().split()))
    A = list(map(int, sys.stdin.readline().strip().split()))
    B = list(map(int, sys.stdin.readline().strip().split()))
    return N, Q, A, B

def solve(N, Q, A, B):
    max_servings = 0
    for i in range(N):
        max_servings_i = min(Q[i] // A[i], Q[i] // B[i])
        max_servings += max_servings_i
    return max_servings

def main():
    N, Q, A, B = read_input()
    max_servings = solve(N, Q, A, B)
    print(max_servings)

if __name__ == "__main__":
    main()