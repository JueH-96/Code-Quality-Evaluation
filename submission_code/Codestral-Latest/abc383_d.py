# YOUR CODE HERE
import math

def count_numbers_with_9_divisors(N):
    count = 0
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            if i * i <= N:
                count += 1
            if i != N // i and N // i <= N:
                count += 1
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])

    result = count_numbers_with_9_divisors(N)
    print(result)

if __name__ == "__main__":
    main()