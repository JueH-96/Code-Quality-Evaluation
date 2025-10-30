def main():
    N = int(input().strip())
    P = list(map(int, input().split()))
    pos = [0] * (N + 1)
    for index, person in enumerate(P):
        pos[person] = index + 1
    
    Q = int(input().strip())
    for _ in range(Q):
        A, B = map(int, input().split())
        if pos[A] < pos[B]:
            print(A)
        else:
            print(B)

if __name__ == "__main__":
    main()