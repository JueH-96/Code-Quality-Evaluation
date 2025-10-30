# YOUR CODE HERE
import sys

def solve():
    S = sys.stdin.readline().strip()

    # Create a new string containing only '2' characters
    # This can be done efficiently using a generator expression and join
    result = "".join(char for char in S if char == '2')

    sys.stdout.write(result + "
")

if __name__ == "__main__":
    solve()