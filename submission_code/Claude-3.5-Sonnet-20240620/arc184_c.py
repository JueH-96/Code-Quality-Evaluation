# YOUR CODE HERE
def get_fold_type(i):
    return 1 if bin(i).count('1') % 2 == 0 else 0

def solve(N, A):
    max_count = 0
    for i in range(1, 2**100 - A[-1]):
        count = sum(get_fold_type(i + a) for a in A)
        max_count = max(max_count, count)
    return max_count

N = int(input())
A = list(map(int, input().split()))

print(solve(N, A))