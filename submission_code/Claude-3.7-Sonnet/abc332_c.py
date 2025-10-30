def solve(N, M, S):
    # Number of clean and dirty shirts of each type
    plain_clean = M
    logo_clean = 0
    plain_dirty = 0
    logo_dirty = 0
    logo_needed = 0
    
    for i in range(N):
        if S[i] == '0':  # No plans, wash all T-shirts
            plain_clean += plain_dirty
            logo_clean += logo_dirty
            plain_dirty = 0
            logo_dirty = 0
        elif S[i] == '1':  # Meal - can use plain or logo T-shirt
            if plain_clean > 0:
                plain_clean -= 1
                plain_dirty += 1
            elif logo_clean > 0:
                logo_clean -= 1
                logo_dirty += 1
            else:
                # Need to "buy" a new logo T-shirt and use it
                logo_needed += 1
                logo_dirty += 1
        else:  # S[i] == '2', Competitive programming - must use logo T-shirt
            if logo_clean > 0:
                logo_clean -= 1
                logo_dirty += 1
            else:
                # Need to "buy" a new logo T-shirt and use it
                logo_needed += 1
                logo_dirty += 1
    
    return logo_needed

def main():
    N, M = map(int, input().split())
    S = input()
    result = solve(N, M, S)
    print(result)

if __name__ == "__main__":
    main()