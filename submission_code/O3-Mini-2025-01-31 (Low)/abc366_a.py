def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    T = int(input_data[1])
    A = int(input_data[2])
    
    # Remaining votes that have not been counted
    remaining = N - (T + A)
    
    # Check if Takahashi is guaranteed to win:
    # Even if Aoki gets all remaining votes, Takahashi still wins if T > A + remaining.
    if T > A + remaining:
        print("Yes")
        return
    
    # Check if Aoki is guaranteed to win:
    # Even if Takahashi gets all remaining votes, Aoki wins if A > T + remaining.
    if A > T + remaining:
        print("Yes")
        return
        
    # Otherwise, the election is not decided yet.
    print("No")

if __name__ == '__main__':
    main()