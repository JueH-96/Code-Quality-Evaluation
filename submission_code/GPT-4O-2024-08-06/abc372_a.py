# YOUR CODE HERE
import sys

def remove_dots(S):
    return S.replace('.', '')

if __name__ == "__main__":
    S = sys.stdin.read().strip()
    result = remove_dots(S)
    print(result)