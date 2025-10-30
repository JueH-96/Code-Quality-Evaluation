# YOUR CODE HERE
def main():
    N = int(input().strip())
    S = input().strip()
    T = input().strip()

    for i in range(N):
        x, y = S[i], T[i]
        if x == y:
            continue
        if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
            continue
        if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
            continue
        print("No")
        return
    print("Yes")

main()