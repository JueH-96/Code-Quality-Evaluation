import sys
from collections import defaultdict

def max_1122_sequence_length(N, A):
    max_length = 0

    for start in range(N):
        count = defaultdict(int)
        valid = True

        for end in range(start, N):
            count[A[end]] += 1

            if count[A[end]] > 2:
                valid = False
                break

            if (end - start + 1) % 2 == 0:
                mid = (start + end) // 2
                if any(A[i] != A[i+1] for i in range(start, mid + 1, 2)):
                    valid = False
                    break

                if any(c != 0 and c != 2 for c in count.values()):
                    valid = False
                    break

                if valid:
                    max_length = max(max_length, end - start + 1)

    return max_length

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = max_1122_sequence_length(N, A)
    print(result)

if __name__ == "__main__":
    main()