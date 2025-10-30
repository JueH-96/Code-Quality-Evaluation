# YOUR CODE HERE
def max_1122_subarray_length(N, A):
    max_len = 0
    for i in range(N):
        count = {}
        current_len = 0
        for j in range(i, N):
            num = A[j]
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
            if count[num] > 2:
                break
            if count[num] == 2:
                current_len += 2
            if current_len % 2 == 0 and all(v == 2 for v in count.values()):
                if current_len > max_len:
                    max_len = current_len
    return max_len

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute the result
result = max_1122_subarray_length(N, A)

# Print the result
print(result)