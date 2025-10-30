def main():
    import sys

    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    count = 0

    while True:
        # Get all positive elements
        positive = [x for x in A if x > 0]
        if len(positive) <= 1:
            break
        # Sort in descending order
        A.sort(reverse=True)
        # Decrease the first two elements by 1
        A[0] -= 1
        A[1] -= 1
        count += 1

    print(count)

if __name__ == "__main__":
    main()