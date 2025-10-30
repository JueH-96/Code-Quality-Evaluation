import sys

def max_pyramid_sequence_length(N, A):
    max_k = 1
    left = 0
    right = 0
    current_k = 1
    while right < N:
        while right < N and A[right] >= current_k:
            if A[right] == current_k:
                current_k += 1
            right += 1
        max_k = max(max_k, current_k - 1)
        if right == N:
            break
        current_k = 1
        left = right
    return max_k

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
print(max_pyramid_sequence_length(N, A))