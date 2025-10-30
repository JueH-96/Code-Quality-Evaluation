# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:2+M]))
    X = []
    index = 2 + M
    for i in range(N):
        X.append(list(map(int, input[index:index+M])))
        index += M

    total_nutrients = [0] * M
    for i in range(N):
        for j in range(M):
            total_nutrients[j] += X[i][j]

    for i in range(M):
        if total_nutrients[i] < A[i]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()