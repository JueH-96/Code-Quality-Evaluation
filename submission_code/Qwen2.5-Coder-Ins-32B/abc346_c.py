# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = list(map(int, input[2:]))

    A_set = set(A)
    missing_sum = 0

    for i in range(1, K + 1):
        if i not in A_set:
            missing_sum += i

    print(missing_sum)

if __name__ == "__main__":
    main()