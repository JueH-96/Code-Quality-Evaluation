import sys

def main():
    input = sys.stdin.read
    data = input().split()

    Q = int(data[0])
    index = 1
    A = []
    results = []

    for _ in range(Q):
        query_type = int(data[index])
        if query_type == 1:
            x = int(data[index + 1])
            A.append(x)
            index += 2
        elif query_type == 2:
            k = int(data[index + 1])
            results.append(A[-k])
            index += 2

    for result in results:
        print(result)

if __name__ == "__main__":
    main()