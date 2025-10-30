# YOUR CODE HERE
import sys

def min_button_presses(S):
    count = 0
    i = 0
    while i < len(S):
        if i < len(S) - 1 and S[i] == '0' and S[i + 1] == '0':
            count += 1
            i += 2
        else:
            count += 1
            i += 1
    return count

if __name__ == "__main__":
    S = input().strip()
    print(min_button_presses(S))