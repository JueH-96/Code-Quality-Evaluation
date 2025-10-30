import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    # Using 1-based index
    prev = [0] * (N + 1)
    last_occurrence = dict()
    for i in range(1, N + 1):
        a = A[i - 1]
        if a in last_occurrence:
            prev[i] = last_occurrence[a]
        else:
            prev[i] = 0
        last_occurrence[a] = i

    next_arr = [N + 1] * (N + 1)
    last_occurrence = dict()
    for i in range(N, 0, -1):
        a = A[i - 1]
        if a in last_occurrence:
            next_arr[i] = last_occurrence[a]
        else:
            next_arr[i] = N + 1
        last_occurrence[a] = i

    total = 0
    for i in range(1, N + 1):
        term = (i - prev[i]) * (next_arr[i] - i)
        total += term
    print(total)

if __name__ == "__main__":
    main()