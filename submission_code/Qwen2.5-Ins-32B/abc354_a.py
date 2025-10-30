# YOUR CODE HERE
import sys

def main():
    H = int(input())
    day = 0
    height = 0
    while height <= H:
        day += 1
        height = 2**day - 1
    print(day)

if __name__ == "__main__":
    main()