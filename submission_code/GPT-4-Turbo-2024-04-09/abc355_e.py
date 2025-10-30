import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
L = int(data[1])
R = int(data[2])

def query(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    response = int(input())
    if response == -1:
        exit()
    return response

def power_of_two(x):
    return (x & (x - 1)) == 0

def solve(N, L, R):
    total_sum = 0
    current_length = 1 << N  # 2^N
    while current_length > 0:
        if L % current_length == 0 and L + current_length - 1 <= R:
            # If the current block fits perfectly within the range
            i = N
            j = L // current_length
            total_sum += query(i, j)
            total_sum %= 100
            L += current_length
        current_length //= 2
        N -= 1
    
    print(f"! {total_sum}")

solve(N, L, R)