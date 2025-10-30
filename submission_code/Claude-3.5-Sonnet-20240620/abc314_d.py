# YOUR CODE HERE
def main():
    N = int(input())
    S = list(input().strip())
    Q = int(input())

    for _ in range(Q):
        t, x, c = input().split()
        t = int(t)
        x = int(x)

        if t == 1:
            S[x-1] = c
        elif t == 2:
            S = [char.lower() for char in S]
        elif t == 3:
            S = [char.upper() for char in S]

    print(''.join(S))

if __name__ == "__main__":
    main()