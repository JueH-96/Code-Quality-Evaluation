import sys

def max_servings():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    Q = list(map(int, data[index:index + N]))
    index += N
    A = list(map(int, data[index:index + N]))
    index += N
    B = list(map(int, data[index:index + N]))

    max_servings = 0

    # Check all possible combinations of servings for dish A and dish B
    for a in range(max(Q) // max(A) + 2):
        for b in range(max(Q) // max(B) + 2):
            valid = True
            for i in range(N):
                if Q[i] < a * A[i] + b * B[i]:
                    valid = False
                    break
            if valid:
                max_servings = max(max_servings, a + b)

    print(max_servings)

max_servings()