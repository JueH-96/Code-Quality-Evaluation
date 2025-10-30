def main():
    A, B, D = map(int, input().split())
    sequence = list(range(A, B + 1, D))
    print(" ".join(map(str, sequence)))

# Do not remove this call
main()