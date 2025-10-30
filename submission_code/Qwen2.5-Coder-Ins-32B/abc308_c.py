# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    people = []
    index = 1
    for i in range(N):
        A = int(input[index])
        B = int(input[index + 1])
        success_rate = A / (A + B)
        people.append((success_rate, -i - 1))  # Use negative index to sort in ascending order for ties
        index += 2
    people.sort(reverse=True)
    result = [-p[1] for p in people]  # Convert back to original indexing
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()