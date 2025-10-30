def main():
    # Read integers from stdin
    N, T, A = map(int, input().split())
    
    # Calculate the number of remaining votes
    R = N - (T + A)
    
    # Check if Takahashi's victory is guaranteed
    if T > A + R:
        print("Yes")
        return
    
    # Check if Aoki's victory is guaranteed
    if A > T + R:
        print("Yes")
        return
    
    # Otherwise, it's not decided yet
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()