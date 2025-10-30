import sys

def solve(L, R):
    result = []
    while L < R:
        # Find the largest power of 2 that is less than or equal to R - L
        i = 0
        while (1 << (i + 1)) <= R - L:
            i += 1

        # Calculate the start and end of the good sequence
        start = L
        end = L + (1 << i)

        # If the end of the good sequence exceeds R, adjust it
        if end > R:
            end = R

        # Add the good sequence to the result
        result.append((start, end))

        # Move to the next sequence
        L = end

    return result

def main():
    L, R = map(int, sys.stdin.readline().split())
    result = solve(L, R)
    print(len(result))
    for start, end in result:
        print(start, end)

if __name__ == "__main__":
    main()