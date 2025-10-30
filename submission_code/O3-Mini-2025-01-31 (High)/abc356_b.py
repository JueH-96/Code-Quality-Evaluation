def main():
    import sys
    input_data = sys.stdin.read().split()
    pos = 0

    # Read N and M
    N = int(input_data[pos])
    pos += 1
    M = int(input_data[pos])
    pos += 1

    # Read the nutrient goals for each of the M types
    goals = [int(input_data[pos + i]) for i in range(M)]
    pos += M

    # Initialize total nutrient intake per type
    totals = [0] * M

    # Read each food's nutrients and add to totals
    for _ in range(N):
        for j in range(M):
            totals[j] += int(input_data[pos])
            pos += 1

    # Check if every nutrient meets its goal
    for j in range(M):
        if totals[j] < goals[j]:
            print("No")
            return

    print("Yes")

main()