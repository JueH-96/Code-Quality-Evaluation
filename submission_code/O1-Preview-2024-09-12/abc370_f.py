# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    total_mass = sum(A)
    max_A = max(A)
    low = max_A
    high = total_mass

    def is_possible(x):
        cnt = 0
        sum_piece = 0
        i = 0
        while i < 2 * N:
            index = i % N
            sum_piece += A[index]
            if sum_piece >= x:
                cnt += 1
                sum_piece = 0
                if cnt >= K and i - (cnt - K) * N <= N:
                    return True
            i += 1
            if i - (cnt - K) * N > N:
                break
        return False

    ans_x = 0
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            ans_x = mid
            low = mid + 1
        else:
            high = mid -1

    x = ans_x

    y = 0
    for i in range(N):
        if A[i] + A[(i+1)%N] < x:
            y +=1

    print(f"{x} {y}")



threading.Thread(target=main).start()