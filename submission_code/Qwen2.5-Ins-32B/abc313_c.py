import sys

def min_operations_to_balance_sequence(N, A):
    A.sort()
    total_operations = 0
    median = A[N // 2]
    for num in A:
        total_operations += abs(num - median)
    return total_operations // 2

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    print(min_operations_to_balance_sequence(N, A))

if __name__ == "__main__":
    main()