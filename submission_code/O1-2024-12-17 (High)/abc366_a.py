def main():
    # Read the integers N (total votes), T (votes for Takahashi), A (votes for Aoki)
    N, T, A = map(int, input().split())
    
    # Calculate the number of remaining votes
    remaining = N - (T + A)
    
    # Check if either candidate has more votes than the opponent can possibly catch up
    if T - A > remaining or A - T > remaining:
        print("Yes")
    else:
        print("No")

# Do not remove the following function call
main()