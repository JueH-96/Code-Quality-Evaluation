# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().splitlines()
    Q = int(input[0])
    A = []
    results = []

    for i in range(1, Q + 1):
        query = input[i].split()
        if query[0] == '1':
            x = int(query[1])
            A.append(x)
        elif query[0] == '2':
            k = int(query[1])
            results.append(A[-k])

    for result in results:
        print(result)

if __name__ == "__main__":
    main()