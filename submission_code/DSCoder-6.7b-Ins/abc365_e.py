def solve():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))

    # Initialize the prefix XOR array
    prefix_xor = [0] * (N + 1)
    for i in range(N):
        prefix_xor[i + 1] = prefix_xor[i] ^ A[i]

    # Initialize the result
    result = 0

    # Calculate the result
    for i in range(N):
        result += (N - i) * (A[i] ^ prefix_xor[i])

    # Print the result
    print(result)

solve()