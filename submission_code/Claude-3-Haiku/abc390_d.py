from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Compute the bitwise XOR of all the bags
    total_xor = 0
    for a in A:
        total_xor ^= a

    # Count the number of possible XOR values
    possible_xors = set()
    possible_xors.add(total_xor)

    for i in range(N):
        for j in range(i+1, N):
            new_xor = total_xor ^ A[i] ^ A[j]
            possible_xors.add(new_xor)

    return len(possible_xors)

print(solve())