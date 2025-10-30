# YOUR CODE HERE
import sys
input = sys.stdin.read

def max_1122_sequence_length(N, A):
    max_length = 0
    for i in range(N):
        count = {}
        for j in range(i, N):
            count[A[j]] = count.get(A[j], 0) + 1
            if count[A[j]] > 2:
                break
            if all(v == 2 for v in count.values()):
                max_length = max(max_length, j - i + 1)
    return max_length

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    print(max_1122_sequence_length(N, A))

if __name__ == "__main__":
    main()