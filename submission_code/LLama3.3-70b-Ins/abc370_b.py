import sys

def read_input():
    """Read input from stdin."""
    lines = sys.stdin.readlines()
    N = int(lines[0].strip())
    A = [list(map(int, line.strip().split())) for line in lines[1:]]
    return N, A

def combine_elements(N, A):
    """Combine elements according to the given rules."""
    current_element = 1
    for i in range(1, N + 1):
        if current_element >= i:
            current_element = A[current_element - 1][i - 1]
        else:
            current_element = A[i - 1][current_element - 1]
    return current_element

def main():
    """Main function."""
    N, A = read_input()
    final_element = combine_elements(N, A)
    print(final_element)

if __name__ == "__main__":
    main()