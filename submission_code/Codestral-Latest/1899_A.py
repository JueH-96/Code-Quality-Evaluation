def determine_winner(n):
    if n % 3 == 0:
        return "First"
    elif (n + 1) % 3 == 0:
        return "Second"
    else:
        return "First"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    t = int(data[0])
    results = []

    for i in range(t):
        n = int(data[i + 1])
        results.append(determine_winner(n))

    for result in results:
        print(result)

if __name__ == "__main__":
    main()