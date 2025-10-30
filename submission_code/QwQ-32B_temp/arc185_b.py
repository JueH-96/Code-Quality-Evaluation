import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        total = sum(A)
        min_sum = A[0] * N
        if total >= min_sum:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()