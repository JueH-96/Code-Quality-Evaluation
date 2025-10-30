# YOUR CODE HERE
import sys

def determine_winner(n):
    if n % 3 == 0:
        return "First"
    if (n + 1) % 3 == 0:
        return "First"
    if (n - 1) % 3 == 0:
        return "Second"
    if (n + 2) % 3 == 0:
        return "Second"
    if (n - 2) % 3 == 0:
        return "First"
    return "Second"

def main():
    import sys
    input = sys.stdin.read().split()
    t = int(input[0])
    results = []
    for i in range(1, t + 1):
        n = int(input[i])
        results.append(determine_winner(n))
    for result in results:
        print(result)

if __name__ == "__main__":
    main()