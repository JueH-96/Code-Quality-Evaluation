import sys

def determine_winner(n):
    # If n is already divisible by 3, Vanya wins on the first move
    if n % 3 == 0:
        return "First"

    # If n modulo 3 is 1 or 2, Vanya can adjust it to be divisible by 3 within 10 moves
    if n % 3 == 1 or n % 3 == 2:
        return "First"

    # If none of the above conditions are met, Vova wins
    return "Second"

def main():
    input = sys.stdin.read()
    data = input.split()

    t = int(data[0])
    results = []

    for i in range(1, t + 1):
        n = int(data[i])
        results.append(determine_winner(n))

    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()