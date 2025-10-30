# YOUR CODE HERE
import sys

def is_extended_abc_string(s):
    if not s:
        return True
    
    a_part = 0
    b_part = 0
    c_part = 0
    
    for char in s:
        if char == 'A' and b_part == 0 and c_part == 0:
            a_part += 1
        elif char == 'B' and a_part > 0 and b_part == 0 and c_part == 0:
            b_part += 1
        elif char == 'C' and a_part > 0 and b_part > 0:
            c_part += 1
        else:
            return False
    
    return a_part > 0 and b_part > 0 and c_part > 0

def main():
    input = sys.stdin.read().strip()
    if is_extended_abc_string(input):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()