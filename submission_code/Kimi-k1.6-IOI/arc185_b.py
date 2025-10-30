import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(data[idx])
        idx += 1
        A = list(map(int, data[idx:idx+N]))
        idx += N
        sum_total = sum(A)
        current_sum = 0
        possible = True
        for i in range(N):
            current_sum += A[i]
            if current_sum * N > sum_total * (i + 1):
                possible = False
                break
        results.append("Yes" if possible else "No")
    print('
'.join(results))

if __name__ == "__main__":
    main()