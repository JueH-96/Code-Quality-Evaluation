# Read input
import sys

def is_11_22_string(N, S):
    if N % 2 == 0:
        return False
    mid = N // 2
    prefix = S[:mid]
    middle = S[mid]
    suffix = S[mid+1:]
    
    if not all(c == '1' for c in prefix):
        return False
    if middle != '/':
        return False
    if not all(c == '2' for c in suffix):
        return False
    return True

def main():
    input = sys.stdin.read().split()
    if len(input) < 2:
        N = int(input[0])
        S = ''
    else:
        N = int(input[0])
        S = input[1]
    if is_11_22_string(N, S):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()