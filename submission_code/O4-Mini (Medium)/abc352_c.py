import sys
import threading

def main():
    data = sys.stdin
    N = int(data.readline().strip())
    A = [0]*N
    B = [0]*N
    sumA = 0
    for i in range(N):
        a,b = map(int, data.readline().split())
        A[i] = a
        B[i] = b
        sumA += a

    # We want to choose the last giant k to maximize: (sum of all A) - A[k] + B[k].
    # That equals sumA - A[i] + B[i].
    ans = 0
    for i in range(N):
        val = sumA - A[i] + B[i]
        if val > ans:
            ans = val

    print(ans)

if __name__ == "__main__":
    main()