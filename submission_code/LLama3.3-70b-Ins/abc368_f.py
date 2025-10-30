import sys

def read_input():
    """Read input from stdin."""
    N = int(input())
    A = list(map(int, input().split()))
    return N, A

def count_powers_of_two(A):
    """Count the number of powers of two in the list."""
    count = 0
    for num in A:
        if (num & (num - 1) == 0) and num != 1:
            count += 1
    return count

def solve(N, A):
    """Solve the problem."""
    count = count_powers_of_two(A)
    if count % 2 == 1:
        return "Anna"
    else:
        return "Bruno"

def main():
    """Main function."""
    N, A = read_input()
    result = solve(N, A)
    print(result)

if __name__ == "__main__":
    main()