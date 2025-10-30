# YOUR CODE HERE
import sys

def check_number(N):
    N_str = str(N)
    if N_str.count('1') == 1 and N_str.count('2') == 2 and N_str.count('3') == 3:
        return "Yes"
    else:
        return "No"

if __name__ == "__main__":
    input_number = int(sys.stdin.read().strip())
    print(check_number(input_number))