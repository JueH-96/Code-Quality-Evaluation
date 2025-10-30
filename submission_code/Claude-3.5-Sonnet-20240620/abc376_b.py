# YOUR CODE HERE
def min_distance(a, b, n):
    return min((b - a) % n, (a - b) % n)

def solve():
    n, q = map(int, input().split())
    left, right = 1, 2
    total_operations = 0

    for _ in range(q):
        hand, target = input().split()
        target = int(target)

        if hand == 'L':
            if target != right:
                total_operations += min_distance(left, target, n)
                left = target
        else:  # hand == 'R'
            if target != left:
                total_operations += min_distance(right, target, n)
                right = target

    print(total_operations)

solve()