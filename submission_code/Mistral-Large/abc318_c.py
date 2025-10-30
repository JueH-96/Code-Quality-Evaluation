import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    D = int(data[1])
    P = int(data[2])
    F = list(map(int, data[3:]))

    # Sort the fares to use the passes on the most expensive days
    F.sort(reverse=True)

    # Calculate the total cost without any passes
    total_cost = sum(F)

    # Calculate the cost with using passes
    for i in range(0, N, D):
        batch_cost = P
        regular_cost = sum(F[i:i+D])
        if batch_cost < regular_cost:
            total_cost -= regular_cost - batch_cost

    print(total_cost)

if __name__ == "__main__":
    main()