def find_winner(N, A):
    # Calculate the nim-sum (XOR) of the number of moves possible for each A[i]
    nim_sum = 0
    for number in A:
        # Count the number of divisors excluding the number itself
        count_divisors = 0
        for d in range(1, number // 2 + 1):
            if number % d == 0:
                count_divisors += 1
        nim_sum ^= count_divisors
    
    # If the nim-sum is zero, Bruno wins; otherwise, Anna wins
    return "Bruno" if nim_sum == 0 else "Anna"

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

print(find_winner(N, A))