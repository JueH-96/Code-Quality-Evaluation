from collections import deque

def solve():
    N, Q = map(int, input().split())
    instructions = []
    for _ in range(Q):
        H, T = input().split()
        T = int(T)
        instructions.append((H, T))

    left, right = 1, 2
    total_ops = 0

    for H, T in instructions:
        if H == 'L':
            if left == T:
                continue
            while left != T:
                if right == left - 1:
                    left -= 1
                else:
                    left += 1
                total_ops += 1
        else:
            if right == T:
                continue
            while right != T:
                if left == right - 1:
                    right += 1
                else:
                    right -= 1
                total_ops += 1

    return total_ops

print(solve())