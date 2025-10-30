# YOUR CODE HERE
def max_1122_sequence(N, A):
    max_length = 0
    current_length = 0
    last_num = None
    seen = set()

    for i in range(N):
        if A[i] == last_num:
            current_length += 1
            if current_length % 2 == 0:
                if A[i] in seen:
                    max_length = max(max_length, current_length)
                else:
                    seen.add(A[i])
        else:
            if current_length % 2 == 1:
                current_length = 0
                seen.clear()
            else:
                current_length = 1
                seen = {A[i]}
        last_num = A[i]

    return max_length

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print output
print(max_1122_sequence(N, A))