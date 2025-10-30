import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # count[i] will track how many times we've seen i so far
    # second_pos[i] will store the position of the middle (2nd) occurrence
    count = [0] * (N + 1)
    second_pos = [0] * (N + 1)

    # Traverse the sequence and record the index of the second occurrence
    for idx, x in enumerate(A, start=1):
        count[x] += 1
        if count[x] == 2:
            second_pos[x] = idx

    # Build a list of (second_occurrence_index, value) and sort by the index
    pairs = [(second_pos[i], i) for i in range(1, N + 1)]
    pairs.sort()  # sorts by the first element, i.e., the second occurrence position

    # Extract the values in sorted order
    result = [str(val) for _, val in pairs]
    print(" ".join(result))

if __name__ == "__main__":
    main()