# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    total_sum = sum(X)
    sum_decrease = 0
    for i in range(N-3):
        A = X[i]
        B = X[i+1]
        C = X[i+2]
        D = X[i+3]
        if A + D < B + C:
            sum_decrease += 2*(B + C - A - D)
    minimal_sum = total_sum - sum_decrease
    print(minimal_sum)

if __name__ == "__main__":
    main()