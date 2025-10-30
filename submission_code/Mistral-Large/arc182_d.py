import sys
input = sys.stdin.read

def min_operations_to_make_sequences_equal(N, M, A, B):
    if A == B:
        return 0

    def is_good_sequence(seq):
        for i in range(N - 1):
            if seq[i] == seq[i + 1]:
                return False
        return True

    def operations_needed(a, b):
        diff = (b - a) % M
        return min(diff, M - diff)

    total_operations = 0
    for i in range(N):
        if A[i] != B[i]:
            # Calculate the minimum operations needed to make A[i] equal to B[i]
            ops = operations_needed(A[i], B[i])
            total_operations += ops

            # Check if the operation makes the sequence invalid
            new_A = A[:]
            new_A[i] = B[i]
            if not is_good_sequence(new_A):
                return -1

            # Update A
            A[i] = B[i]

    return total_operations

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:]))

    result = min_operations_to_make_sequences_equal(N, M, A, B)
    print(result)

if __name__ == "__main__":
    main()