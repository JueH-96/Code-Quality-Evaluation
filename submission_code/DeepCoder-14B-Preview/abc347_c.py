def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = int(input[1])
    B = int(input[2])
    D = list(map(int, input[3:3+N]))
    
    M = A + B
    if M == 0:
        print("No")
        return
    
    a = []
    b = []
    for d in D:
        ai = (-d) % M
        bi = (ai + A - 1) % M
        a.append(ai)
        b.append(bi)
    
    max_a = max(a)
    min_b = min(b)
    
    if max_a <= min_b:
        print("Yes")
    else:
        # Check if the wrapping case is possible
        if (max_a <= M - 1) and (min_b >= 0) and (max_a <= min_b + M):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()