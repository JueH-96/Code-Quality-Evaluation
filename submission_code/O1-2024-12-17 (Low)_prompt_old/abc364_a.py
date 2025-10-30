def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    dishes = data[1:]
    
    # Iterate through dishes checking for consecutive sweets.
    # If consecutive sweets occur at positions i and i+1,
    # and i+1 is NOT the last dish, then Takahashi cannot finish.
    for i in range(1, N):
        if dishes[i] == "sweet" and dishes[i-1] == "sweet":
            # If this occurs before the last dish, answer is "No"
            # Otherwise (i == N-1), we have finished all dishes.
            if i < N - 1:
                print("No")
                return
    
    # If no condition prevents finishing, print "Yes"
    print("Yes")

# Call solve() to execute
solve()