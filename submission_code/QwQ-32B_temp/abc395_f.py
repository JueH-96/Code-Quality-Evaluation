import sys

def main():
    N, X = map(int, sys.stdin.readline().split())
    U = []
    D = []
    min_sum = float('inf')
    for _ in range(N):
        u, d = map(int, sys.stdin.readline().split())
        U.append(u)
        D.append(d)
        current_sum = u + d
        if current_sum < min_sum:
            min_sum = current_sum
    H_max = min_sum

    left = 0
    right = H_max
    best_H = 0

    while left <= right:
        mid = (left + right) // 2
        possible = True
        a0 = max(mid - D[0], 0)
        b0 = min(U[0], mid)
        if a0 > b0:
            possible = False
        else:
            low = a0
            high = b0
            for i in range(1, N):
                a_i = max(mid - D[i], 0)
                b_i = min(U[i], mid)
                new_low = max(a_i, low - X)
                new_high = min(b_i, high + X)
                if new_low > new_high:
                    possible = False
                    break
                low, high = new_low, new_high
        if possible:
            best_H = mid
            left = mid + 1
        else:
            right = mid - 1

    total = sum(u + d for u, d in zip(U, D))
    print(total - best_H * N)

if __name__ == "__main__":
    main()