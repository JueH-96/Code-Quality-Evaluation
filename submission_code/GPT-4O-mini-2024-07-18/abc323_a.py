# YOUR CODE HERE
def check_even_positions(S):
    for i in range(2, 17, 2):  # Check even indices from 2 to 16
        if S[i - 1] != '0':  # Convert to 0-based index
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    check_even_positions(S)