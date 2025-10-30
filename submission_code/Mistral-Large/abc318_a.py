import sys

def count_full_moons(N, M, P):
    # Calculate the number of full moons Takahashi can see between day 1 and day N inclusive.
    count = 0
    current_day = M

    while current_day <= N:
        count += 1
        current_day += P

    return count

def main():
    # Read input from stdin
    input = sys.stdin.read()
    N, M, P = map(int, input.split())

    # Calculate the result
    result = count_full_moons(N, M, P)

    # Write the result to stdout
    print(result)

if __name__ == "__main__":
    main()