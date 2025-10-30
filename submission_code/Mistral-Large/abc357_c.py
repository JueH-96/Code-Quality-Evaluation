import sys

def generate_carpet(K):
    if K == 0:
        return ["#"]

    smaller_carpet = generate_carpet(K - 1)
    size = 3 ** K
    third = size // 3

    top_bottom = [
        smaller_carpet[i % third] + "###" + smaller_carpet[i % third]
        for i in range(third)
    ]

    middle = [
        smaller_carpet[i] + "..." + smaller_carpet[i]
        for i in range(third)
    ]

    return top_bottom + middle + top_bottom

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    carpet = generate_carpet(N)
    for line in carpet:
        print(line)

if __name__ == "__main__":
    main()