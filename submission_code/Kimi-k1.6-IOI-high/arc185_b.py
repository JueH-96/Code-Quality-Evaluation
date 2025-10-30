import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    T = int(data[ptr])
    ptr += 1
    for _ in range(T):
        N = int(data[ptr])
        ptr += 1
        A = list(map(int, data[ptr:ptr+N]))
        ptr += N
        sum_total = sum(A)
        current_sum = 0
        possible = True
        for i in range(N):
            current_sum += A[i]
            if current_sum * N > sum_total * (i + 1):
                possible = False
                break
        print("Yes" if possible else "No")

if __name__ == "__main__":
    main()