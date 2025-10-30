import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Calculate the target average value
    total_sum = sum(A)
    target_avg = (total_sum + N - 1) // N  # This is equivalent to ceil(total_sum / N)

    # Calculate the number of operations needed
    operations = 0
    for value in A:
        if value > target_avg:
            operations += (value - target_avg)

    print(operations)

if __name__ == "__main__":
    solve()