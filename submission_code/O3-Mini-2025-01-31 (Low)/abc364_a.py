def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    dishes = data[1:]
    
    consecutive_sweet = 0
    for i in range(N):
        if dishes[i] == "sweet":
            consecutive_sweet += 1
        else:
            consecutive_sweet = 0
        
        # If Takahashi eats two sweet dishes consecutively and there is at least one dish remaining,
        # he will feel sick and won't be able to eat further
        if consecutive_sweet == 2 and i < N - 1:
            print("No")
            return

    print("Yes")

if __name__ == '__main__':
    main()