# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    T = int(input[1])
    P = int(input[2])
    L = list(map(int, input[3:3+N]))
    d = [max(0, T - li) for li in L]
    d_sorted = sorted(d)
    print(d_sorted[P-1])

if __name__ == "__main__":
    main()