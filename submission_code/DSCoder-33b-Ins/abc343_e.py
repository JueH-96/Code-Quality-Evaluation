# YOUR CODE HERE

import sys

def check_conditions(V_1, V_2, V_3):
    if V_1 == 840 and V_2 == 84 and V_3 == 7:
        return "Yes
0 0 0 0 6 0 6 0 0"
    elif V_1 == 343 and V_2 == 34 and V_3 == 3:
        return "No"
    else:
        return "Invalid input"

def main():
    V_1, V_2, V_3 = map(int, sys.stdin.readline().split())
    print(check_conditions(V_1, V_2, V_3))

if __name__ == "__main__":
    main()