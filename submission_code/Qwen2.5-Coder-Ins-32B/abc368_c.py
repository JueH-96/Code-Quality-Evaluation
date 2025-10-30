# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:]))

    T = 0
    for h in H:
        while h > 0:
            T += 1
            if T % 3 == 0:
                h -= 3
            else:
                h -= 1

    print(T)

if __name__ == "__main__":
    main()