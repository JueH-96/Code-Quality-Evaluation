# YOUR CODE HERE
def max_servings():
    N = int(input())
    Q = list(map(int, input().split()))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    def can_make(a, b):
        return all((a * A[i] + b * B[i] <= Q[i]) for i in range(N))

    left, right = 0, sum(Q) + 1
    while right - left > 1:
        mid = (left + right) // 2
        possible = False
        for a in range(mid + 1):
            b = mid - a
            if can_make(a, b):
                possible = True
                break
        if possible:
            left = mid
        else:
            right = mid

    print(left)

max_servings()