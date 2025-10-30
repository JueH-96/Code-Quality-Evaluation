import sys

def count_valid_positions(N, A):
    count = 0
    for i in range(1, N + 1):
        # Find the positions of the two occurrences of color i
        first_index = A.index(i)
        second_index = A.index(i, first_index + 1)

        # Check if there is exactly one person between them
        if abs(first_index - second_index) == 2:
            count += 1

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = count_valid_positions(N, A)
    print(result)

if __name__ == "__main__":
    main()