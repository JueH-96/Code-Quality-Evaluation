def find_missing_numbers(N, M, A):
    # Create a set of all numbers from 1 to N
    all_numbers = set(range(1, N + 1))
    # Create a set from the given list A
    present_numbers = set(A)
    # Find the missing numbers by subtracting present from all
    missing_numbers = sorted(all_numbers - present_numbers)
    
    # Prepare the output
    C = len(missing_numbers)
    return C, missing_numbers

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2 + M]))
    
    C, missing_numbers = find_missing_numbers(N, M, A)
    
    # Print the results
    print(C)
    if C > 0:
        print(" ".join(map(str, missing_numbers)))