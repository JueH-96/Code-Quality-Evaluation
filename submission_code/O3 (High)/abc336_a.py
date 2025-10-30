import sys

def main():
    # Read N from standard input
    N_line = sys.stdin.readline().strip()
    if not N_line:
        return
    N = int(N_line)
    
    # Construct the Dragon String: 'L' + N * 'o' + 'n' + 'g'
    dragon_string = 'L' + 'o' * N + 'n' + 'g'
    
    # Output the result
    print(dragon_string)

# Call main to execute the program
if __name__ == "__main__":
    main()