N = int(input())
S = input().strip()

if N % 2 == 0:
    print("No")
else:
    mid = (N + 1) // 2
    if S[mid - 1] != '/':
        print("No")
    else:
        # Check characters before '/'
        if not all(S[i] == '1' for i in range(mid - 1)):
            print("No")
        else:
            # Check characters after '/'
            if not all(S[i] == '2' for i in range(mid, N)):
                print("No")
            else:
                print("Yes")