# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    giants = [(int(input[2*i+1]), int(input[2*i+2])) for i in range(N)]
    giants.sort(key=lambda x: (x[1] - x[0], x[1]))
    current_height = 0
    for A, B in giants:
        current_height += A
        current_height = max(current_height, B)
    print(current_height)

if __name__ == "__main__":
    main()