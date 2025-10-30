import sys

def nand(a, b):
    return 0 if a == 1 and b == 1 else 1

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    A = [int(x) for x in S]
    total_sum = 0

    for i in range(N):
        current_value = A[i]
        total_sum += current_value
        for j in range(i + 1, N):
            current_value = nand(current_value, A[j])
            total_sum += current_value

    print(total_sum)

if __name__ == "__main__":
    main()