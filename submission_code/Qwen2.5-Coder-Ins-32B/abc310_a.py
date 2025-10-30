# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    P = int(input[1])
    Q = int(input[2])
    D = list(map(int, input[3:]))

    min_cost = P
    for d in D:
        min_cost = min(min_cost, Q + d)

    print(min_cost)

if __name__ == "__main__":
    main()