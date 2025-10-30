def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    S = input().strip()

    # We'll process consecutive days between laundry days (S[i]=='0') as segments.
    # In a segment, Takahashi cannot reuse a T-shirt.
    # He has M plain shirts available fresh at the beginning of each segment.
    # For each day:
    # - Meal days ('1'): can use either a plain shirt (if available) or a logo shirt.
    # - CP event days ('2'): must use a logo shirt.
    #
    # In a segment, let meal_count be the count of meal days and comp_count be the count of CP-event days.
    # Plain shirts can be used for at most M meal days, so extra meal days (if meal_count > M)
    # require using logo shirts.
    # Thus, in that segment, the number of logo shirts needed equals:
    #    comp_count + max(0, meal_count - M)
    #
    # Since the T-shirts are washed at the end of each segment (day with no plan '0'),
    # the logo shirts purchased can be reused in different segments.
    # Therefore, the minimum number of logo shirts that need to be purchased overall
    # is the maximum over segments of (comp_count + max(0, meal_count - M)).
    
    max_logo_needed = 0
    meal_count = 0
    comp_count = 0
    
    for ch in S:
        if ch == '0':
            # Laundry day: wash all previously worn shirts so we end a segment.
            logo_needed = comp_count + max(0, meal_count - M)
            max_logo_needed = max(max_logo_needed, logo_needed)
            meal_count = 0
            comp_count = 0
        else:
            if ch == '1':
                meal_count += 1
            elif ch == '2':
                comp_count += 1
    # Process the last segment (if the schedule does not end with '0')
    logo_needed = comp_count + max(0, meal_count - M)
    max_logo_needed = max(max_logo_needed, logo_needed)
    
    print(max_logo_needed)

if __name__ == '__main__':
    main()