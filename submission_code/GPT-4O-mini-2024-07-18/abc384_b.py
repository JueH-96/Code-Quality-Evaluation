def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line for N and R
    first_line = data[0].split()
    N = int(first_line[0])
    R = int(first_line[1])
    
    # Process each contest
    for i in range(1, N + 1):
        contest_info = data[i].split()
        D_i = int(contest_info[0])
        A_i = int(contest_info[1])
        
        # Determine if the rating should be updated
        if D_i == 1 and 1600 <= R <= 2799:
            R += A_i
        elif D_i == 2 and 1200 <= R <= 2399:
            R += A_i
    
    # Output the final rating
    print(R)

if __name__ == "__main__":
    main()