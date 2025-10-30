# YOUR CODE HERE
import sys

def main():
    H = int(sys.stdin.read().strip())
    height = 0
    day = 0
    while height <= H:
        height = (1 << day) - 1
        day += 1
    print(day - 1)

if __name__ == "__main__":
    main()