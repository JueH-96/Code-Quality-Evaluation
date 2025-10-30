# YOUR CODE HERE
def check_number(N):
    # Convert the number to a string to easily count digits
    N_str = str(N)
    
    # Check the conditions
    if N_str.count('1') == 1 and N_str.count('2') == 2 and N_str.count('3') == 3:
        return "Yes"
    else:
        return "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    result = check_number(N)
    print(result)