import sys

def solve():
    N, Q = map(int, input().split())
    left, right = 1, 2
    total_ops = 0

    for _ in range(Q):
        hand, target = input().split()
        target = int(target)

        if hand == 'L':
            if target != right:
                while left != target:
                    if left > target:
                        left -= 1
                        right -= 1
                        if right == 0:
                            right = N
                    else:
                        left += 1
                        right += 1
                        if left > N:
                            left = 1
                    total_ops += 1
            else:
                while left != target:
                    if left > target:
                        left -= 1
                        if right > 1:
                            right -= 1
                        else:
                            right = N
                    else:
                        left += 1
                        if right < N:
                            right += 1
                        else:
                            right = 1
                    total_ops += 1
        else:
            if target != left:
                while right != target:
                    if right > target:
                        right -= 1
                        left -= 1
                        if left == 0:
                            left = N
                    else:
                        right += 1
                        left += 1
                        if right > N:
                            right = 1
                    total_ops += 1
            else:
                while right != target:
                    if right > target:
                        right -= 1
                        if left > 1:
                            left -= 1
                        else:
                            left = N
                    else:
                        right += 1
                        if left < N:
                            left += 1
                        else:
                            left = 1
                    total_ops += 1

    print(total_ops)

if __name__ == "__main__":
    solve()