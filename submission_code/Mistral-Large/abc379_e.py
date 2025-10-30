import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    total_sum = 0
    current_sum = 0
    power_of_ten = 1

    for i in range(N - 1, -1, -1):
        current_sum += int(S[i]) * power_of_ten
        total_sum += current_sum
        power_of_ten = (power_of_ten * 10 + 1) % (10**9 + 7)

    print(total_sum)

solve()