from itertools import accumulate

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Calculate prefix XORs
    prefix_xor = list(accumulate(A, lambda x, y: x ^ y))

    # Initialize the answer
    ans = 0

    # Calculate the answer using the prefix XORs
    for i in range(N):
        for j in range(i + 1, N):
            ans += prefix_xor[j] ^ prefix_xor[i]

    print(ans)

solve()