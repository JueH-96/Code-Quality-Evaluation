# YOUR CODE HERE
import sys

def main():
    import sys

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    R = int(data[1])
    index = 2
    for _ in range(N):
        D = int(data[index])
        A = int(data[index + 1])
        index += 2
        if D == 1:
            if 1600 <= R <= 2799:
                R += A
        elif D == 2:
            if 1200 <= R <= 2399:
                R += A
    print(R)

if __name__ == "__main__":
    main()