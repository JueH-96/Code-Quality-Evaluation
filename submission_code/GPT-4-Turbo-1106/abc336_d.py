def max_pyramid_size(N, A):
    left = 0
    right = N - 1
    height = 0
    max_height = 0

    while left <= right:
        if A[left] < A[right]:
            if A[left] > height:
                height += 1
            else:
                height = A[left]
            left += 1
        else:
            if A[right] > height:
                height += 1
            else:
                height = A[right]
            right -= 1
        max_height = max(max_height, height)

    return max_height

# Read input from stdin
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Solve the problem and print the result to stdout
print(max_pyramid_size(N, A))