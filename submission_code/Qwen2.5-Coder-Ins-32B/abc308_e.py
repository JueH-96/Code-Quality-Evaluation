# YOUR CODE HERE
import sys

def mex(a, b, c):
    for i in range(3):
        if i != a and i != b and i != c:
            return i

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    S = input[N+1]

    count = 0
    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                if S[i] + S[j] + S[k] == "MEX":
                    count += mex(A[i], A[j], A[k])
    print(count)

if __name__ == "__main__":
    main()