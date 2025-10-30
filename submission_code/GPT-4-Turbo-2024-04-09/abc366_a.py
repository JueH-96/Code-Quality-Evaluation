def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    A = int(data[2])
    
    remaining_votes = N - (T + A)
    
    if T + remaining_votes <= A:
        print("No")
    elif A + remaining_votes < T:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()