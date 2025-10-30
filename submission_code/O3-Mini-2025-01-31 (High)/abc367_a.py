def main():
    A, B, C = map(int, input().split())
    
    # When sleep time does NOT cross midnight: B < C
    if B < C:
        if B <= A < C:
            print("No")
        else:
            print("Yes")
    # When sleep time crosses midnight: B > C
    else:
        if A >= B or A < C:
            print("No")
        else:
            print("Yes")

if __name__ == '__main__':
    main()