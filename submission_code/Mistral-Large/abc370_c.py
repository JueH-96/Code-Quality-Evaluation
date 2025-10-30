import sys

def find_min_operations(S, T):
    n = len(S)
    operations = []

    for i in range(n):
        if S[i] != T[i]:
            # Change each character from S[i] to T[i]
            for j in range(ord(S[i]) + 1, ord(T[i]) + 1):
                S = S[:i] + chr(j) + S[i+1:]
                operations.append(S)

    return operations

def main():
    input = sys.stdin.read
    data = input().split()

    S = data[0]
    T = data[1]

    operations = find_min_operations(S, T)

    print(len(operations))
    for op in operations:
        print(op)

if __name__ == "__main__":
    main()