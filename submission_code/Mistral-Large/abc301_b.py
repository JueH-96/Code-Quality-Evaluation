import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

def process_sequence(A):
    i = 0
    while i < len(A) - 1:
        if abs(A[i] - A[i + 1]) != 1:
            if A[i] < A[i + 1]:
                insert_sequence = list(range(A[i] + 1, A[i + 1]))
            else:
                insert_sequence = list(range(A[i] - 1, A[i + 1], -1))
            A = A[:i + 1] + insert_sequence + A[i + 1:]
            i = 0  # Restart the check from the beginning
        else:
            i += 1
    return A

result = process_sequence(A)
print(" ".join(map(str, result)))