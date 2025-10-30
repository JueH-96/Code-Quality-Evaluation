# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    Q = list(map(int, input[1:N+1]))
    A = list(map(int, input[N+1:2*N+1]))
    B = list(map(int, input[2*N+1:3*N+1]))

    def can_make(s):
        for i in range(N):
            if A[i] * s + B[i] * (s - a) > Q[i]:
                return False
        return True

    max_servings = 0
    for a in range(1000001):
        b = 0
        while a + b <= 1000000 and can_make(a + b):
            b += 1
        max_servings = max(max_servings, a + b - 1)

    print(max_servings)

if __name__ == "__main__":
    main()