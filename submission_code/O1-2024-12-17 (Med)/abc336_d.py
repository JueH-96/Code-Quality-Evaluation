def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # inc[i] will store the length of the largest contiguous "ascending by 1 from 1..x"
    # subarray ending exactly at index i (0-based).
    inc = [0]*N
    inc[0] = 1  # Since A[0] >= 1 by constraints, we can at least form "1"
    for i in range(1, N):
        if A[i] >= inc[i-1] + 1:
            inc[i] = inc[i-1] + 1
        else:
            inc[i] = 1  # start a new subarray of length 1 if A[i] >= 1 (always true)

    # dec[i] will store the length of the largest contiguous "ascending by 1 from 1..x"
    # subarray starting exactly at index i (0-based), when read left-to-right from i.
    # Equivalently, this is the descending-part length from i towards the right in the original array.
    dec = [0]*N
    dec[N-1] = 1
    for i in range(N-2, -1, -1):
        if A[i] >= dec[i+1] + 1:
            dec[i] = dec[i+1] + 1
        else:
            dec[i] = 1

    # The maximum pyramid size occurs at some apex i, and is min(inc[i], dec[i]).
    # We take the maximum of that over all i.
    answer = 0
    for i in range(N):
        candidate = min(inc[i], dec[i])
        if candidate > answer:
            answer = candidate

    print(answer)

# Do not forget to call main()
main()