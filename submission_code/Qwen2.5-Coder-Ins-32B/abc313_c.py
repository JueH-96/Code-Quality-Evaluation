# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    min_val = min(A)
    max_val = max(A)

    if max_val - min_val <= 1:
        print(0)
        return

    count = 0
    # Try to make all elements equal to min_val or min_val + 1
    for num in A:
        if num > min_val + 1:
            count += num - (min_val + 1)
        elif num < min_val:
            count += min_val - num

    print(count)

if __name__ == "__main__":
    main()