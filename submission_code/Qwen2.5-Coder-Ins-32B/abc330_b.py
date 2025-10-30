# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    L = int(input[1])
    R = int(input[2])
    A = list(map(int, input[3:]))

    result = []
    for a in A:
        if a <= L:
            result.append(L)
        elif a >= R:
            result.append(R)
        else:
            result.append(a)

    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()