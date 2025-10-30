# YOUR CODE HERE
import sys

def count_expressible_numbers(N):
    expressible = set()
    a = 2
    while a * a <= N:
        b = 2
        while (value := a ** b) <= N:
            expressible.add(value)
            b += 1
        a += 1
    return len(expressible)

def main():
    N = int(sys.stdin.read().strip())
    print(count_expressible_numbers(N))

if __name__ == "__main__":
    main()