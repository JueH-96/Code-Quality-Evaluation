import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    # Initialize the expected value array
    expected_value = [0] * (N + 1)

    # Calculate the expected value in reverse order
    for i in range(N, 0, -1):
        # The expected value at each step is the sum of the expected values of all possible next steps
        expected_value[i-1] = (expected_value[i] * (N - i) + A[i-1]) / N

    # The expected value for x=0 is the expected value at the first step
    result = expected_value[0]

    # Convert the result to an integer modulo 998244353
    result = int(result * (N + 1) % MOD)

    print(result)

if __name__ == "__main__":
    main()