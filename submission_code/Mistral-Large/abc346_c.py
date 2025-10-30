import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))

    # Use a set to track the numbers that appear in A
    seen = set(A)

    # Calculate the sum of all numbers from 1 to K that are not in A
    missing_sum = sum(x for x in range(1, K + 1) if x not in seen)

    print(missing_sum)

if __name__ == "__main__":
    main()